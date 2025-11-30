"""Session management commands.

This module provides CLI commands for session lifecycle management:
- create: Create a new session
- list: List sessions with optional state filter
- info: Get detailed session information
- delete: Delete a session
- restore: Restore a session from backup

All commands use the service layer for business logic and abstract
JSON-RPC protocol details.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
import json

from agentcore_cli.container import get_session_service
from agentcore_cli.services.exceptions import (
    ValidationError,
    SessionNotFoundError,
    OperationError,
    ServiceError,
)

app = typer.Typer(
    name="session",
    help="Manage session lifecycle and state",
    no_args_is_help=True,
)

console = Console()


@app.command()
def create(
    name: Annotated[str, typer.Option("--name", "-n", help="Session name")],
    context: Annotated[
        str | None,
        typer.Option(
            "--context",
            "-c",
            help="Session context as JSON string (optional)",
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
    """Create a new session.

    This command creates a new session with the specified name and optional
    context. Sessions maintain state across multiple operations.

    Examples:
        # Create a simple session
        agentcore session create --name analysis-session

        # Create with context
        agentcore session create -n test-session -c '{"user": "alice"}'

        # Get JSON output
        agentcore session create -n my-session --json
    """
    # Parse context if provided (do this before try block)
    context_dict = None
    if context:
        try:
            context_dict = json.loads(context)
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON in context:[/red] {str(e)}")
            raise typer.Exit(2)

    try:
        # Get service from DI container
        service = get_session_service()

        # Call service method
        session_id = service.create(
            name=name,
            context=context_dict,
        )

        # Format output
        if json_output:
            result = {
                "session_id": session_id,
                "name": name,
                "context": context_dict,
            }
            print(json.dumps(result, indent=2))
        else:
            console.print(f"[green]✓[/green] Session created successfully")
            console.print(f"[bold]Session ID:[/bold] {session_id}")
            console.print(f"[bold]Name:[/bold] {name}")
            if context_dict:
                console.print(f"[bold]Context:[/bold]")
                print(json.dumps(context_dict, indent=2))

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
    state: Annotated[
        str | None,
        typer.Option(
            "--state",
            "-s",
            help="Filter by state (active, inactive, archived)",
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-l",
            help="Maximum number of sessions to return",
        ),
    ] = 100,
    offset: Annotated[
        int,
        typer.Option(
            "--offset",
            "-o",
            help="Number of sessions to skip",
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
    """List sessions with optional filtering.

    Examples:
        # List all sessions
        agentcore session list

        # Filter by state
        agentcore session list --state active

        # Limit results
        agentcore session list --limit 10

        # Pagination
        agentcore session list --limit 10 --offset 20

        # Get JSON output
        agentcore session list --json
    """
    try:
        # Get service from DI container
        service = get_session_service()

        # Call service method
        sessions = service.list_sessions(state=state, limit=limit, offset=offset)

        # Format output
        if json_output:
            print(json.dumps(sessions, indent=2))
        else:
            if not sessions:
                console.print("[yellow]No sessions found[/yellow]")
                return

            # Create table
            table = Table(title=f"Sessions ({len(sessions)})")
            table.add_column("Session ID", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("State", style="yellow")
            table.add_column("Created", style="blue")

            for session in sessions:
                session_id = session.get("session_id", "N/A")
                name = session.get("name", "N/A")
                state_val = session.get("state", "N/A")
                created = session.get("created_at", "N/A")

                # Truncate name if too long
                if len(name) > 40:
                    name = name[:37] + "..."

                table.add_row(session_id, name, state_val, created)

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
    session_id: Annotated[str, typer.Argument(help="Session identifier")],
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output in JSON format",
        ),
    ] = False,
) -> None:
    """Get detailed information about a session.

    Examples:
        # Get session info
        agentcore session info session-001

        # Get JSON output
        agentcore session info session-001 --json
    """
    try:
        # Get service from DI container
        service = get_session_service()

        # Call service method
        session = service.get(session_id)

        # Format output
        if json_output:
            print(json.dumps(session, indent=2))
        else:
            console.print(f"[bold]Session Information[/bold]")
            console.print(f"[bold]ID:[/bold] {session.get('session_id', 'N/A')}")
            console.print(f"[bold]Name:[/bold] {session.get('name', 'N/A')}")
            console.print(f"[bold]State:[/bold] {session.get('state', 'N/A')}")
            console.print(f"[bold]Created:[/bold] {session.get('created_at', 'N/A')}")
            console.print(f"[bold]Updated:[/bold] {session.get('updated_at', 'N/A')}")

            # Show context if present
            context = session.get("context")
            if context:
                console.print(f"[bold]Context:[/bold]")
                print(json.dumps(context, indent=2))

    except ValidationError as e:
        console.print(f"[red]Validation error:[/red] {e.message}")
        raise typer.Exit(2)
    except SessionNotFoundError as e:
        console.print(f"[red]Session not found:[/red] {e.message}")
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
def delete(
    session_id: Annotated[str, typer.Argument(help="Session identifier")],
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Force deletion even if session is active",
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
    """Delete a session.

    Examples:
        # Delete a session
        agentcore session delete session-001

        # Force deletion
        agentcore session delete session-001 --force

        # Get JSON output
        agentcore session delete session-001 --json
    """
    try:
        # Get service from DI container
        service = get_session_service()

        # Call service method
        success = service.delete(session_id, force=force)

        # Format output
        if json_output:
            result = {"success": success, "session_id": session_id}
            print(json.dumps(result, indent=2))
        else:
            if success:
                console.print(f"[green]✓[/green] Session deleted successfully")
                console.print(f"[bold]Session ID:[/bold] {session_id}")
            else:
                console.print(f"[red]Failed to delete session {session_id}[/red]")
                raise typer.Exit(1)

    except ValidationError as e:
        console.print(f"[red]Validation error:[/red] {e.message}")
        raise typer.Exit(2)
    except SessionNotFoundError as e:
        console.print(f"[red]Session not found:[/red] {e.message}")
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
def restore(
    session_id: Annotated[str, typer.Argument(help="Session identifier")],
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output in JSON format",
        ),
    ] = False,
) -> None:
    """Restore a session from backup.

    This command restores a session's context from a previous backup,
    allowing you to continue from a saved state.

    Examples:
        # Restore a session
        agentcore session restore session-001

        # Get JSON output
        agentcore session restore session-001 --json
    """
    try:
        # Get service from DI container
        service = get_session_service()

        # Call service method
        context = service.restore(session_id)

        # Format output
        if json_output:
            result = {"session_id": session_id, "context": context}
            print(json.dumps(result, indent=2))
        else:
            console.print(f"[green]✓[/green] Session restored successfully")
            console.print(f"[bold]Session ID:[/bold] {session_id}")

            if context:
                console.print(f"[bold]Restored Context:[/bold]")
                print(json.dumps(context, indent=2))
            else:
                console.print("[yellow]No context data in session[/yellow]")

    except ValidationError as e:
        console.print(f"[red]Validation error:[/red] {e.message}")
        raise typer.Exit(2)
    except SessionNotFoundError as e:
        console.print(f"[red]Session not found:[/red] {e.message}")
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
