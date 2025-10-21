"""Agent management commands for AgentCore CLI."""

from __future__ import annotations

import json
import sys
from typing import Annotated, Optional

import typer
from rich.console import Console

from agentcore_cli.client import AgentCoreClient
from agentcore_cli.config import Config
from agentcore_cli.exceptions import (
    AgentCoreError,
    AuthenticationError,
    ConnectionError as CliConnectionError,
)
from agentcore_cli.formatters import (
    format_agent_info,
    format_error,
    format_json,
    format_success,
    format_table,
    format_warning,
)

app = typer.Typer(
    name="agent",
    help="Manage agent lifecycle and discovery",
    no_args_is_help=True,
)

console = Console()


@app.command()
def register(
    name: Annotated[
        str,
        typer.Option("--name", "-n", help="Agent name (required)"),
    ],
    capabilities: Annotated[
        str,
        typer.Option("--capabilities", "-c", help="Comma-separated capabilities (required)"),
    ],
    cost_per_request: Annotated[
        float,
        typer.Option("--cost-per-request", help="Cost per request in USD"),
    ] = 0.01,
    requirements: Annotated[
        Optional[str],
        typer.Option("--requirements", "-r", help="JSON string of requirements"),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output in JSON format"),
    ] = False,
) -> None:
    """Register a new agent with AgentCore.

    Example:
        agentcore agent register --name "code-analyzer" --capabilities "python,analysis"

    With requirements:
        agentcore agent register \\
            --name "code-analyzer" \\
            --capabilities "python,analysis,linting" \\
            --requirements '{"memory": "512MB", "cpu": "0.5"}' \\
            --cost-per-request 0.01
    """
    try:
        # Load configuration
        config = Config.load()

        # Parse capabilities
        cap_list = [c.strip() for c in capabilities.split(",") if c.strip()]
        if not cap_list:
            console.print(format_error("At least one capability is required"))
            sys.exit(2)

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
        result = client.call("agent.register", {
            "name": name,
            "capabilities": cap_list,
            "cost_per_request": cost_per_request,
            "requirements": req_dict,
        })

        # Output result
        if json_output:
            console.print(format_json(result))
        else:
            agent_id = result.get("agent_id", "N/A")
            console.print(format_success(f"Agent registered: {agent_id}"))
            console.print(f"  Name: {name}")
            console.print(f"  Capabilities: {', '.join(cap_list)}")

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
def list_agents(
    status: Annotated[
        Optional[str],
        typer.Option("--status", "-s", help="Filter by status (active, inactive, etc.)"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", help="Maximum number of agents to display"),
    ] = 100,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output in JSON format"),
    ] = False,
) -> None:
    """List all registered agents.

    Example:
        agentcore agent list

    Filter by status:
        agentcore agent list --status active

    JSON output:
        agentcore agent list --json
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
        if status:
            params["status"] = status

        # Call API
        result = client.call("agent.list", params)

        # Extract agents list
        agents = result.get("agents", [])

        # Output result
        if json_output:
            console.print(format_json(agents))
        else:
            if not agents:
                console.print("[dim]No agents found[/dim]")
                return

            # Display as table
            columns = ["agent_id", "name", "status", "capabilities"]
            table_output = format_table(agents, columns=columns, title="Registered Agents")
            console.print(table_output)

            # Show count
            console.print(f"\n[dim]Total: {len(agents)} agent(s)[/dim]")

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
    agent_id: Annotated[
        str,
        typer.Argument(help="Agent ID to retrieve information for"),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output in JSON format"),
    ] = False,
) -> None:
    """Get detailed information about a specific agent.

    Example:
        agentcore agent info agent-12345

    JSON output:
        agentcore agent info agent-12345 --json
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
        result = client.call("agent.info", {"agent_id": agent_id})

        # Output result
        if json_output:
            console.print(format_json(result))
        else:
            agent_info = format_agent_info(result)
            console.print(agent_info)

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
def remove(
    agent_id: Annotated[
        str,
        typer.Argument(help="Agent ID to remove"),
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
    """Remove an agent from AgentCore.

    Example:
        agentcore agent remove agent-12345

    Skip confirmation:
        agentcore agent remove agent-12345 --force
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

        # Get agent info for confirmation
        if not force and not json_output:
            try:
                agent_info = client.call("agent.info", {"agent_id": agent_id})
                agent_name = agent_info.get("name", agent_id)

                console.print(format_warning(
                    f"Remove agent '{agent_name}' ({agent_id})?\n"
                    "   This action cannot be undone."
                ))

                confirm = typer.confirm("Continue?", default=False)
                if not confirm:
                    console.print("[yellow]Operation cancelled[/yellow]")
                    sys.exit(0)
            except typer.Exit:
                # Re-raise typer.Exit to maintain exit code
                raise
            except AgentCoreError:
                # Agent might not exist, proceed anyway
                pass

        # Call API
        result = client.call("agent.remove", {"agent_id": agent_id})

        # Output result
        if json_output:
            console.print(format_json(result))
        else:
            console.print(format_success(f"Agent removed: {agent_id}"))

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
def search(
    capability: Annotated[
        list[str],
        typer.Option("--capability", "-c", help="Capability to search for (can specify multiple)"),
    ],
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", help="Maximum number of agents to return"),
    ] = 100,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output in JSON format"),
    ] = False,
) -> None:
    """Search for agents by capability.

    Example:
        agentcore agent search --capability python

    Multiple capabilities:
        agentcore agent search --capability python --capability testing

    JSON output:
        agentcore agent search --capability python --json
    """
    if not capability:
        console.print(format_error("At least one capability is required"))
        console.print("Use: agentcore agent search --capability <name>")
        sys.exit(2)

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
        result = client.call("agent.search", {
            "capabilities": capability,
            "limit": limit,
        })

        # Extract agents
        agents = result.get("agents", [])

        # Output result
        if json_output:
            console.print(format_json(agents))
        else:
            if not agents:
                caps_str = ", ".join(capability)
                console.print(f"[dim]No agents found with capabilities: {caps_str}[/dim]")
                return

            # Display as table
            columns = ["agent_id", "name", "status", "capabilities"]
            caps_str = ", ".join(capability)
            table_output = format_table(
                agents,
                columns=columns,
                title=f"Agents with capabilities: {caps_str}"
            )
            console.print(table_output)

            # Show count
            console.print(f"\n[dim]Total: {len(agents)} agent(s)[/dim]")

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
