"""Agent management commands.

This module provides CLI commands for agent lifecycle management:
- register: Register a new agent
- list: List registered agents
- info: Get agent information
- remove: Remove an agent
- search: Search agents by capability

All commands use the service layer for business logic and abstract
JSON-RPC protocol details.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
import json

from agentcore_cli.container import get_agent_service
from agentcore_cli.services.exceptions import (
    ValidationError,
    AgentNotFoundError,
    OperationError,
    ServiceError,
)

app = typer.Typer(
    name="agent",
    help="Manage agent lifecycle and discovery",
    no_args_is_help=True,
)

console = Console()


@app.command()
def register(
    name: Annotated[str, typer.Option("--name", "-n", help="Agent name (must be unique)")],
    capabilities: Annotated[
        str,
        typer.Option(
            "--capabilities",
            "-c",
            help="Comma-separated list of agent capabilities (e.g., 'python,analysis')",
        ),
    ],
    endpoint_url: Annotated[
        str | None,
        typer.Option(
            "--endpoint-url",
            "-e",
            help="Agent endpoint URL (e.g., 'http://localhost:5000'). If not provided, uses a default placeholder.",
        ),
    ] = None,
    cost_per_request: Annotated[
        float,
        typer.Option(
            "--cost-per-request",
            "-r",
            help="Cost per request in dollars",
        ),
    ] = 0.01,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output in JSON format",
        ),
    ] = False,
) -> None:
    """Register a new agent with AgentCore.

    This command registers a new agent with the specified capabilities.
    The agent will be available for task assignment and discovery.

    Examples:
        # Register a simple agent
        agentcore agent register --name analyzer --capabilities python,analysis

        # Register with endpoint URL
        agentcore agent register -n executor -c python,execution -e http://localhost:5000

        # Register with custom cost
        agentcore agent register -n executor -c python,execution -r 0.05

        # Get JSON output
        agentcore agent register -n tester -c testing,qa --json
    """
    try:
        # Parse capabilities (split by comma and strip whitespace)
        cap_list = [c.strip() for c in capabilities.split(",") if c.strip()]

        # Get service from DI container
        service = get_agent_service()

        # Call service method
        agent_id = service.register(
            name=name,
            capabilities=cap_list,
            endpoint_url=endpoint_url,
            cost_per_request=cost_per_request,
        )

        # Format output
        if json_output:
            result = {
                "id": agent_id,  # For backward compatibility with tests
                "agent_id": agent_id,
                "name": name,
                "capabilities": cap_list,
                "cost_per_request": cost_per_request,
            }
            # Use print() for JSON to avoid Rich's text wrapping
            print(json.dumps(result, indent=2))
        else:
            console.print(f"[green]✓[/green] Agent registered successfully")
            console.print(f"[bold]Agent ID:[/bold] {agent_id}")
            console.print(f"[bold]Name:[/bold] {name}")
            console.print(f"[bold]Capabilities:[/bold] {', '.join(cap_list)}")

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
            help="Filter by status (active, inactive, error)",
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-l",
            help="Maximum number of agents to return",
        ),
    ] = 100,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output in JSON format",
        ),
    ] = False,
) -> None:
    """List registered agents.

    Examples:
        # List all agents
        agentcore agent list

        # Filter by status
        agentcore agent list --status active

        # Limit results
        agentcore agent list --limit 10

        # Get JSON output
        agentcore agent list --json
    """
    try:
        # Get service from DI container
        service = get_agent_service()

        # Call service method
        agents = service.list_agents(status=status, limit=limit)

        # Format output
        if json_output:
            # Use print() for JSON to avoid Rich's text wrapping
            print(json.dumps(agents, indent=2))
        else:
            if not agents:
                console.print("[yellow]No agents found[/yellow]")
                return

            # Create table
            table = Table(title=f"Registered Agents ({len(agents)})")
            table.add_column("Agent ID", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Status", style="yellow")
            table.add_column("Capabilities", style="blue")

            for agent in agents:
                agent_id = agent.get("agent_id", "N/A")
                name = agent.get("name", "N/A")
                status_val = agent.get("status", "N/A")
                caps = agent.get("capabilities", [])
                caps_str = ", ".join(caps) if caps else "N/A"

                table.add_row(agent_id, name, status_val, caps_str)

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
    agent_id: Annotated[str, typer.Argument(help="Agent identifier")],
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output in JSON format",
        ),
    ] = False,
) -> None:
    """Get detailed information about an agent.

    Examples:
        # Get agent info
        agentcore agent info agent-001

        # Get JSON output
        agentcore agent info agent-001 --json
    """
    try:
        # Get service from DI container
        service = get_agent_service()

        # Call service method
        agent = service.get(agent_id)

        # Format output
        if json_output:
            # Use print() for JSON to avoid Rich's text wrapping
            print(json.dumps(agent, indent=2))
        else:
            console.print(f"[bold]Agent Information[/bold]")
            console.print(f"[bold]ID:[/bold] {agent.get('agent_id', 'N/A')}")
            console.print(f"[bold]Name:[/bold] {agent.get('name', 'N/A')}")
            console.print(f"[bold]Status:[/bold] {agent.get('status', 'N/A')}")
            console.print(
                f"[bold]Capabilities:[/bold] {', '.join(agent.get('capabilities', []))}"
            )
            console.print(
                f"[bold]Cost per request:[/bold] ${agent.get('cost_per_request', 0)}"
            )

    except ValidationError as e:
        console.print(f"[red]Validation error:[/red] {e.message}")
        raise typer.Exit(2)
    except AgentNotFoundError as e:
        console.print(f"[red]Agent not found:[/red] {e.message}")
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
def remove(
    agent_id: Annotated[str, typer.Argument(help="Agent identifier")],
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Force removal even if agent is active",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip confirmation prompt",
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
    """Remove an agent.

    Examples:
        # Remove an agent
        agentcore agent remove agent-001

        # Force removal
        agentcore agent remove agent-001 --force

        # Skip confirmation
        agentcore agent remove agent-001 --yes

        # Get JSON output
        agentcore agent remove agent-001 --json
    """
    try:
        # Confirm removal unless --yes is provided
        if not yes and not json_output:
            confirm = typer.confirm(f"Are you sure you want to remove agent '{agent_id}'?")
            if not confirm:
                console.print("[yellow]Operation cancelled[/yellow]")
                raise typer.Exit(0)

        # Get service from DI container
        service = get_agent_service()

        # Call service method
        success = service.remove(agent_id, force=force)

        # Format output
        if json_output:
            result = {"success": success, "agent_id": agent_id}
            # Use print() for JSON to avoid Rich's text wrapping
            print(json.dumps(result, indent=2))
        else:
            if success:
                console.print(f"[green]✓[/green] Agent removed successfully")
                console.print(f"[bold]Agent ID:[/bold] {agent_id}")
            else:
                console.print(f"[red]Failed to remove agent {agent_id}[/red]")
                raise typer.Exit(1)

    except ValidationError as e:
        console.print(f"[red]Validation error:[/red] {e.message}")
        raise typer.Exit(2)
    except AgentNotFoundError as e:
        console.print(f"[red]Agent not found:[/red] {e.message}")
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
def search(
    capability: Annotated[
        str,
        typer.Option(
            "--capability",
            "-c",
            help="Capability to search for",
        ),
    ],
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-l",
            help="Maximum number of agents to return",
        ),
    ] = 100,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output in JSON format",
        ),
    ] = False,
) -> None:
    """Search agents by capability.

    Examples:
        # Search for agents with Python capability
        agentcore agent search --capability python

        # Limit results
        agentcore agent search -c analysis --limit 10

        # Get JSON output
        agentcore agent search -c testing --json
    """
    try:
        # Get service from DI container
        service = get_agent_service()

        # Call service method
        agents = service.search(capability=capability, limit=limit)

        # Format output
        if json_output:
            # Use print() for JSON to avoid Rich's text wrapping
            print(json.dumps(agents, indent=2))
        else:
            if not agents:
                console.print(f"[yellow]No agents found with capability '{capability}'[/yellow]")
                return

            # Create table
            table = Table(title=f"Agents with '{capability}' capability ({len(agents)})")
            table.add_column("Agent ID", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Status", style="yellow")
            table.add_column("Capabilities", style="blue")

            for agent in agents:
                agent_id = agent.get("agent_id", "N/A")
                name = agent.get("name", "N/A")
                status_val = agent.get("status", "N/A")
                caps = agent.get("capabilities", [])
                caps_str = ", ".join(caps) if caps else "N/A"

                table.add_row(agent_id, name, status_val, caps_str)

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
