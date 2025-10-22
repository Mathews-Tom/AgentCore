"""Session management commands for AgentCore CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
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
    name="session",
    help="Save and resume long-running workflows",
    no_args_is_help=True,
)

console = Console()


@app.command()
def save(
    name: Annotated[
        str,
        typer.Option("--name", "-n", help="Session name (required)"),
    ],
    description: Annotated[
        str,
        typer.Option("--description", "-d", help="Session description"),
    ] = "",
    tags: Annotated[
        Optional[list[str]],
        typer.Option("--tag", "-t", help="Session tags (can be specified multiple times)"),
    ] = None,
    metadata: Annotated[
        Optional[str],
        typer.Option("--metadata", "-m", help="JSON string of custom metadata"),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output in JSON format"),
    ] = False,
) -> None:
    """Save current workflow session.

    Example:
        agentcore session save --name "feature-dev" --description "Building auth"

    With tags:
        agentcore session save \\
            --name "feature-dev" \\
            --tag "auth" \\
            --tag "backend"

    With metadata:
        agentcore session save \\
            --name "feature-dev" \\
            --metadata '{"branch": "main", "sprint": "S1"}'
    """
    try:
        # Load configuration
        config = Config.load()

        # Parse metadata if provided
        metadata_dict = {}
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
                if not isinstance(metadata_dict, dict):
                    console.print(format_error("Metadata must be a JSON object"))
                    sys.exit(2)
            except json.JSONDecodeError as e:
                console.print(format_error(f"Invalid JSON in metadata: {e}"))
                sys.exit(2)

        # Build session parameters
        params: dict[str, object] = {
            "name": name,
            "description": description,
            "tags": tags or [],
            "metadata": metadata_dict,
        }

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
                progress.add_task(description="Saving session...", total=None)
                result = client.call("session.save", params)
        else:
            result = client.call("session.save", params)

        # Output result
        if json_output:
            console.print(format_json(result))
        else:
            session_id = result.get("session_id", "N/A")
            console.print(format_success(f"Session saved: {session_id}"))
            console.print(f"  Name: {name}")
            if description:
                console.print(f"  Description: {description}")
            if tags:
                console.print(f"  Tags: {', '.join(tags)}")
            console.print(f"\n  Resume with: agentcore session resume {session_id}")

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
    session_id: Annotated[
        str,
        typer.Argument(help="Session ID to resume"),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output in JSON format"),
    ] = False,
) -> None:
    """Resume a saved session.

    Example:
        agentcore session resume session-12345
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
                task = progress.add_task(description="Restoring session...", total=100)
                result = client.call("session.resume", {"session_id": session_id})
                progress.update(task, completed=100)
        else:
            result = client.call("session.resume", {"session_id": session_id})

        # Output result
        if json_output:
            console.print(format_json(result))
        else:
            console.print(format_success(f"Session resumed: {session_id}"))
            tasks_count = result.get("tasks_count", 0)
            agents_count = result.get("agents_count", 0)
            console.print(f"  Tasks restored: {tasks_count}")
            console.print(f"  Agents restored: {agents_count}")

            # Show session details if available
            if "name" in result:
                console.print(f"  Name: {result['name']}")
            if "description" in result and result["description"]:
                console.print(f"  Description: {result['description']}")

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
def list_sessions(
    status_filter: Annotated[
        Optional[str],
        typer.Option("--status", "-s", help="Filter by status (active, paused, completed)"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", help="Maximum number of sessions to display"),
    ] = 100,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output in JSON format"),
    ] = False,
) -> None:
    """List sessions.

    Example:
        agentcore session list

    Filter by status:
        agentcore session list --status active

    JSON output:
        agentcore session list --json
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
        result = client.call("session.list", params)

        # Extract sessions list
        sessions = result.get("sessions", [])

        # Output result
        if json_output:
            console.print(format_json(sessions))
        else:
            if not sessions:
                console.print("[dim]No sessions found[/dim]")
                return

            # Display as table
            columns = ["session_id", "name", "status", "tasks_count", "created_at"]
            table_output = format_table(sessions, columns=columns, title="Sessions")
            console.print(table_output)

            # Show count
            console.print(f"\n[dim]Total: {len(sessions)} session(s)[/dim]")

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
def info(
    session_id: Annotated[
        str,
        typer.Argument(help="Session ID to show details for"),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output in JSON format"),
    ] = False,
) -> None:
    """Get session details.

    Example:
        agentcore session info session-12345

    JSON output:
        agentcore session info session-12345 --json
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
        result = client.call("session.info", {"session_id": session_id})

        # Output result
        if json_output:
            console.print(format_json(result))
        else:
            _display_session_info(result)

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
def delete(
    session_id: Annotated[
        str,
        typer.Argument(help="Session ID to delete"),
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
    """Delete a session.

    Example:
        agentcore session delete session-12345

    Skip confirmation:
        agentcore session delete session-12345 --force
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

        # Get session info for confirmation
        if not force and not json_output:
            try:
                session_info = client.call("session.info", {"session_id": session_id})
                session_name = session_info.get("name", session_id)
                session_status = session_info.get("status", "unknown")

                console.print(format_warning(
                    f"Delete session '{session_name}' ({session_id})?\n"
                    f"   Current status: {session_status}\n"
                    "   This action cannot be undone."
                ))

                confirm = typer.confirm("Continue?", default=False)
                if not confirm:
                    console.print("[yellow]Operation cancelled[/yellow]")
                    sys.exit(0)
            except typer.Exit:
                raise
            except AgentCoreError:
                # Session might not exist, proceed anyway
                pass

        # Call API
        result = client.call("session.delete", {"session_id": session_id})

        # Output result
        if json_output:
            console.print(format_json(result))
        else:
            console.print(format_success(f"Session deleted: {session_id}"))

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
def export(
    session_id: Annotated[
        str,
        typer.Argument(help="Session ID to export"),
    ],
    output_file: Annotated[
        str,
        typer.Option("--output", "-o", help="Output file path (required)"),
    ],
    pretty: Annotated[
        bool,
        typer.Option("--pretty", "-p", help="Pretty-print JSON output"),
    ] = True,
) -> None:
    """Export session for debugging.

    Example:
        agentcore session export session-12345 --output session.json

    Without pretty-printing:
        agentcore session export session-12345 --output session.json --no-pretty
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
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(description="Exporting session...", total=None)
            result = client.call("session.export", {"session_id": session_id})

        # Save to file
        output_path = Path(output_file)
        with open(output_path, "w") as f:
            if pretty:
                json.dump(result, f, indent=2, sort_keys=True)
            else:
                json.dump(result, f)

        console.print(format_success(f"Session exported to: {output_path.absolute()}"))

        # Show file size
        file_size = output_path.stat().st_size
        if file_size < 1024:
            size_str = f"{file_size} bytes"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"
        console.print(f"  File size: {size_str}")

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

def _display_session_info(session_data: dict[str, object]) -> None:
    """Display session info in detailed view.

    Args:
        session_data: Session data dictionary
    """
    console.print(f"[bold]Session ID:[/bold] {session_data.get('session_id', 'N/A')}")
    console.print(f"[bold]Name:[/bold] {session_data.get('name', 'N/A')}")

    # Status with color
    status = str(session_data.get("status", "unknown"))
    status_colored = _format_status_value(status)
    console.print(f"[bold]Status:[/bold] {status_colored}")

    # Description
    description = session_data.get("description")
    if description:
        console.print(f"[bold]Description:[/bold] {description}")

    # Tags
    tags = session_data.get("tags", [])
    if tags:
        console.print(f"[bold]Tags:[/bold] {', '.join(tags)}")

    # Counts
    tasks_count = session_data.get("tasks_count", 0)
    agents_count = session_data.get("agents_count", 0)
    console.print(f"[bold]Tasks:[/bold] {tasks_count}")
    console.print(f"[bold]Agents:[/bold] {agents_count}")

    # Timestamps
    created_at = session_data.get("created_at")
    if created_at:
        console.print(f"[bold]Created:[/bold] {created_at}")

    updated_at = session_data.get("updated_at")
    if updated_at:
        console.print(f"[bold]Updated:[/bold] {updated_at}")

    # Metadata
    metadata = session_data.get("metadata")
    if metadata and isinstance(metadata, dict) and metadata:
        console.print("\n[bold]Metadata:[/bold]")
        for key, value in metadata.items():
            console.print(f"  {key}: {value}")

    # Tasks summary
    tasks = session_data.get("tasks", [])
    if tasks:
        console.print("\n[bold]Tasks:[/bold]")
        task_table = Table(show_header=True)
        task_table.add_column("Task ID", style="cyan")
        task_table.add_column("Type")
        task_table.add_column("Status")

        for task in tasks[:10]:  # Show first 10 tasks
            task_id = str(task.get("task_id", "N/A"))
            task_type = str(task.get("type", "N/A"))
            task_status = str(task.get("status", "unknown"))
            task_table.add_row(task_id, task_type, _format_status_value(task_status))

        console.print(task_table)

        if len(tasks) > 10:
            console.print(f"[dim]  ... and {len(tasks) - 10} more task(s)[/dim]")


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
    if status_lower in ("active", "running", "pending"):
        return f"[yellow]{status}[/yellow]"
    if status_lower in ("failed", "error"):
        return f"[red]{status}[/red]"
    if status_lower in ("paused", "cancelled"):
        return f"[dim]{status}[/dim]"
    return status
