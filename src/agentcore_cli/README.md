# AgentCore CLI

Developer-friendly command-line interface for AgentCore.

## Overview

The AgentCore CLI provides a Python-based command-line interface that wraps the AgentCore JSON-RPC 2.0 API with familiar command patterns (similar to docker, kubectl, git).

## Installation

### Using uv (recommended)

```bash
uv add agentcore
```

### Using pip

```bash
pip install agentcore
```

## Usage

### Basic Commands

```bash
# Show help
agentcore --help

# Show version
agentcore --version
```

## Development

### Project Structure

```
src/agentcore_cli/
├── __init__.py          # Package metadata
├── main.py              # Main CLI entry point
├── commands/            # Command groups
│   └── __init__.py
└── utils/               # CLI utilities
    └── __init__.py
```

### Running Tests

```bash
# Run CLI tests
uv run pytest tests/cli/ -v

# Run with coverage
uv run pytest tests/cli/ --cov=src/agentcore_cli --cov-report=term-missing
```

### Type Checking

```bash
uv run mypy src/agentcore_cli/
```

## Technology Stack

- **CLI Framework:** Typer (type-based CLI framework)
- **Output Formatting:** Rich (beautiful terminal output)
- **Configuration:** PyYAML (YAML config files)
- **Validation:** Pydantic (input/output validation)

## Future Commands

The CLI will support the following command groups in upcoming releases:

- `agentcore agent` - Agent lifecycle management
- `agentcore task` - Task creation and monitoring
- `agentcore session` - Session management
- `agentcore workflow` - Workflow orchestration
- `agentcore config` - Configuration management

## License

MIT License - see LICENSE file for details.
