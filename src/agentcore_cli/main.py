"""Main CLI entry point for AgentCore."""

from __future__ import annotations

import sys
from typing import Annotated

import typer
from rich.console import Console

from agentcore_cli import __version__
from agentcore_cli.commands import agent, config, session, task, workflow

app = typer.Typer(
    name="agentcore",
    help="AgentCore CLI - Developer-friendly command-line interface for AgentCore",
    no_args_is_help=True,
    add_completion=False,
)

# Register command groups
app.add_typer(agent.app, name="agent")
app.add_typer(task.app, name="task")
app.add_typer(session.app, name="session")
app.add_typer(workflow.app, name="workflow")
app.add_typer(config.app, name="config")

console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"AgentCore CLI version: {__version__}")
        raise typer.Exit(0)


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """
    AgentCore CLI - Command-line interface for AgentCore.

    A developer-friendly CLI that wraps the AgentCore JSON-RPC 2.0 API
    with familiar command patterns (docker, kubectl, git).

    Use 'agentcore COMMAND --help' for help with specific commands.
    """
    pass


def cli_main() -> None:
    """Entry point for the CLI."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
