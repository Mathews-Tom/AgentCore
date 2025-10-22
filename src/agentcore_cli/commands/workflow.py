"""Workflow management commands.

This module provides CLI commands for workflow execution and monitoring:
- run: Run a workflow from YAML definition
- list: List workflows with optional status filter
- info: Get detailed workflow information
- stop: Stop a running workflow

All commands use the service layer for business logic and abstract
JSON-RPC protocol details.
"""

from __future__ import annotations

from typing import Annotated
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
import json
import yaml

from agentcore_cli.container import get_workflow_service
from agentcore_cli.services.exceptions import (
    ValidationError,
    WorkflowNotFoundError,
    OperationError,
    ServiceError,
)

app = typer.Typer(
    name="workflow",
    help="Manage workflow execution and monitoring",
    no_args_is_help=True,
)

console = Console()


@app.command()
def run(
    workflow_file: Annotated[Path, typer.Argument(help="Path to workflow YAML file")],
    parameters: Annotated[
        str | None,
        typer.Option(
            "--parameters",
            "-p",
            help="Workflow parameters as JSON string (optional)",
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
    """Run a workflow from YAML definition.

    This command loads a workflow definition from a YAML file and executes it.
    Optional parameters can be provided as a JSON string.

    Examples:
        # Run a workflow
        agentcore workflow run workflow.yaml

        # Run with parameters
        agentcore workflow run workflow.yaml -p '{"repo": "foo/bar"}'

        # Get JSON output
        agentcore workflow run workflow.yaml --json
    """
    # Load and parse YAML file (do this before try block)
    if not workflow_file.exists():
        console.print(f"[red]Error:[/red] Workflow file not found: {workflow_file}")
        raise typer.Exit(1)

    try:
        with open(workflow_file, "r") as f:
            definition = yaml.safe_load(f)
    except yaml.YAMLError as e:
        console.print(f"[red]Invalid YAML:[/red] {str(e)}")
        raise typer.Exit(2)
    except Exception as e:
        console.print(f"[red]Error reading file:[/red] {str(e)}")
        raise typer.Exit(1)

    # Parse parameters if provided
    params_dict = None
    if parameters:
        try:
            params_dict = json.loads(parameters)
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON in parameters:[/red] {str(e)}")
            raise typer.Exit(2)

    try:
        # Get service from DI container
        service = get_workflow_service()

        # Call service method
        workflow_id = service.run(
            definition=definition,
            parameters=params_dict,
        )

        # Format output
        if json_output:
            result = {
                "workflow_id": workflow_id,
                "definition": definition,
                "parameters": params_dict,
            }
            console.print(json.dumps(result, indent=2))
        else:
            console.print(f"[green]✓[/green] Workflow started successfully")
            console.print(f"[bold]Workflow ID:[/bold] {workflow_id}")
            console.print(f"[bold]Name:[/bold] {definition.get('name', 'N/A')}")
            if params_dict:
                console.print(f"[bold]Parameters:[/bold]")
                console.print(json.dumps(params_dict, indent=2))

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
            help="Filter by status (running, completed, failed, cancelled)",
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-l",
            help="Maximum number of workflows to return",
        ),
    ] = 100,
    offset: Annotated[
        int,
        typer.Option(
            "--offset",
            "-o",
            help="Number of workflows to skip",
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
    """List workflows with optional filtering.

    Examples:
        # List all workflows
        agentcore workflow list

        # Filter by status
        agentcore workflow list --status running

        # Limit results
        agentcore workflow list --limit 10

        # Pagination
        agentcore workflow list --limit 10 --offset 20

        # Get JSON output
        agentcore workflow list --json
    """
    try:
        # Get service from DI container
        service = get_workflow_service()

        # Call service method
        workflows = service.list_workflows(status=status, limit=limit, offset=offset)

        # Format output
        if json_output:
            console.print(json.dumps(workflows, indent=2))
        else:
            if not workflows:
                console.print("[yellow]No workflows found[/yellow]")
                return

            # Create table
            table = Table(title=f"Workflows ({len(workflows)})")
            table.add_column("Workflow ID", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Status", style="yellow")
            table.add_column("Created", style="blue")

            for workflow in workflows:
                workflow_id = workflow.get("workflow_id", "N/A")
                name = workflow.get("name", "N/A")
                status_val = workflow.get("status", "N/A")
                created = workflow.get("created_at", "N/A")

                # Truncate name if too long
                if len(name) > 40:
                    name = name[:37] + "..."

                table.add_row(workflow_id, name, status_val, created)

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
    workflow_id: Annotated[str, typer.Argument(help="Workflow identifier")],
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output in JSON format",
        ),
    ] = False,
) -> None:
    """Get detailed information about a workflow.

    Examples:
        # Get workflow info
        agentcore workflow info workflow-001

        # Get JSON output
        agentcore workflow info workflow-001 --json
    """
    try:
        # Get service from DI container
        service = get_workflow_service()

        # Call service method
        workflow = service.get(workflow_id)

        # Format output
        if json_output:
            console.print(json.dumps(workflow, indent=2))
        else:
            console.print(f"[bold]Workflow Information[/bold]")
            console.print(f"[bold]ID:[/bold] {workflow.get('workflow_id', 'N/A')}")
            console.print(f"[bold]Name:[/bold] {workflow.get('name', 'N/A')}")
            console.print(f"[bold]Status:[/bold] {workflow.get('status', 'N/A')}")
            console.print(f"[bold]Created:[/bold] {workflow.get('created_at', 'N/A')}")
            console.print(f"[bold]Updated:[/bold] {workflow.get('updated_at', 'N/A')}")

            # Show definition if present
            definition = workflow.get("definition")
            if definition:
                console.print(f"[bold]Definition:[/bold]")
                console.print(json.dumps(definition, indent=2))

            # Show parameters if present
            parameters = workflow.get("parameters")
            if parameters:
                console.print(f"[bold]Parameters:[/bold]")
                console.print(json.dumps(parameters, indent=2))

            # Show steps if present
            steps = workflow.get("steps")
            if steps:
                console.print(f"[bold]Steps:[/bold]")
                console.print(json.dumps(steps, indent=2))

    except ValidationError as e:
        console.print(f"[red]Validation error:[/red] {e.message}")
        raise typer.Exit(2)
    except WorkflowNotFoundError as e:
        console.print(f"[red]Workflow not found:[/red] {e.message}")
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
def stop(
    workflow_id: Annotated[str, typer.Argument(help="Workflow identifier")],
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Force stop even if workflow is in critical state",
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
    """Stop a running workflow.

    Examples:
        # Stop a workflow
        agentcore workflow stop workflow-001

        # Force stop
        agentcore workflow stop workflow-001 --force

        # Get JSON output
        agentcore workflow stop workflow-001 --json
    """
    try:
        # Get service from DI container
        service = get_workflow_service()

        # Call service method
        success = service.stop(workflow_id, force=force)

        # Format output
        if json_output:
            result = {"success": success, "workflow_id": workflow_id}
            console.print(json.dumps(result, indent=2))
        else:
            if success:
                console.print(f"[green]✓[/green] Workflow stopped successfully")
                console.print(f"[bold]Workflow ID:[/bold] {workflow_id}")
            else:
                console.print(f"[red]Failed to stop workflow {workflow_id}[/red]")
                raise typer.Exit(1)

    except ValidationError as e:
        console.print(f"[red]Validation error:[/red] {e.message}")
        raise typer.Exit(2)
    except WorkflowNotFoundError as e:
        console.print(f"[red]Workflow not found:[/red] {e.message}")
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
