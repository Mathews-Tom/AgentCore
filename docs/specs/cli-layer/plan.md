# CLI Layer Implementation Plan

**Component:** CLI Layer
**Timeline:** 4 weeks (2 sprints)
**Team:** 1 senior Python developer
**Total Effort:** 34 story points
**Created:** 2025-09-30

---

## Executive Summary

The CLI Layer provides a developer-friendly command-line interface for AgentCore, significantly lowering the barrier to entry while maintaining full access to enterprise capabilities. This thin wrapper around the JSON-RPC 2.0 API follows familiar CLI patterns (docker, kubectl) and delivers time-to-first-task reduction from hours to minutes.

**Business Value:**

- 3x increase in developer adoption (target metric)
- Reduced onboarding time from days to hours
- Improved developer experience without compromising architecture
- Competitive parity with tools like Claude-Flow

**Technical Approach:**

- Python Click/Typer framework for robust CLI
- Thin wrapper pattern (no business logic in CLI)
- Multi-level configuration (CLI → env → file → defaults)
- Rich output formatting for human readability

---

## Phase Overview

### Sprint 1: Core Framework (18 SP)

**Duration:** Week 1-2
**Goal:** Working CLI with basic agent and task commands
**Deliverables:**

- CLI framework setup
- JSON-RPC client implementation
- Configuration management
- Agent and task commands

### Sprint 2: Advanced Features (16 SP)

**Duration:** Week 3-4
**Goal:** Session management, workflow support, polished UX
**Deliverables:**

- Session commands
- Workflow commands
- Multiple output formats
- Interactive prompts
- Testing and documentation

---

## Sprint 1: Core Framework (Week 1-2, 18 SP)

### Week 1: Foundation

#### Days 1-2: Project Setup (3 SP)

**Tasks:**

- Initialize Python package structure
- Choose CLI framework (Click vs Typer evaluation)
- Configure build system (pyproject.toml)
- Set up development environment
- Configure testing framework (pytest)

**Deliverables:**

```text
agentcore-cli/
├── pyproject.toml
├── README.md
├── src/
│   └── agentcore_cli/
│       ├── __init__.py
│       ├── cli.py
│       ├── client.py
│       ├── config.py
│       └── formatters.py
├── tests/
│   ├── __init__.py
│   ├── test_cli.py
│   └── test_client.py
└── .github/
    └── workflows/
        └── test.yml
```

**Technology Decision:**

```python
# Option 1: Click (mature, widely used)
import click

@click.group()
def cli():
    pass

@cli.command()
@click.option('--name', required=True)
def register(name):
    pass

# Option 2: Typer (modern, type-based)
import typer

app = typer.Typer()

@app.command()
def register(name: str):
    pass
```

**Recommendation:** Typer (better type safety, automatic help generation, modern syntax)

#### Days 3-4: JSON-RPC Client (5 SP)

**Tasks:**

- Implement JSON-RPC 2.0 client class
- Add request/response handling
- Implement retry logic with exponential backoff
- Add connection pooling
- Error translation (JSON-RPC errors → user-friendly messages)

**Implementation:**

```python
# src/agentcore_cli/client.py
import requests
from typing import Dict, Any, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class AgentCoreClient:
    """JSON-RPC 2.0 client for AgentCore API"""

    def __init__(
        self,
        api_url: str,
        timeout: int = 30,
        retries: int = 3,
        verify_ssl: bool = True
    ):
        self.api_url = f"{api_url}/api/v1/jsonrpc"
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.session = self._create_session(retries)
        self.request_id = 0

    def _create_session(self, retries: int) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def call(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make JSON-RPC 2.0 call"""
        self.request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self.request_id
        }

        try:
            response = self.session.post(
                self.api_url,
                json=payload,
                timeout=self.timeout,
                verify=self.verify_ssl
            )
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                raise JsonRpcError(data["error"])

            return data.get("result", {})

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to AgentCore API: {e}")
```

**Testing Strategy:**

```python
# tests/test_client.py
import pytest
from unittest.mock import Mock, patch
from agentcore_cli.client import AgentCoreClient

@patch('agentcore_cli.client.requests.Session.post')
def test_successful_call(mock_post):
    mock_post.return_value = Mock(
        json=lambda: {"jsonrpc": "2.0", "result": {"success": True}, "id": 1},
        status_code=200
    )

    client = AgentCoreClient("http://localhost:8001")
    result = client.call("agent.list")

    assert result == {"success": True}
    assert mock_post.call_count == 1
```

#### Days 5: Configuration Management (5 SP)

**Tasks:**

- Implement configuration file loading (YAML)
- Support multi-level config (global, project, env)
- Implement precedence logic
- Add config validation with Pydantic
- Create `config init` command

**Configuration Schema:**

```python
# src/agentcore_cli/config.py
from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, Literal
import os
import yaml

class ApiConfig(BaseModel):
    url: HttpUrl = Field(default="http://localhost:8001")
    timeout: int = Field(default=30, ge=1, le=300)
    retries: int = Field(default=3, ge=0, le=10)
    verify_ssl: bool = True

class AuthConfig(BaseModel):
    type: Literal["jwt", "api_key", "none"] = "none"
    token: Optional[str] = None
    api_key: Optional[str] = None

class OutputConfig(BaseModel):
    format: Literal["json", "table", "tree"] = "table"
    color: bool = True
    timestamps: bool = False
    verbose: bool = False

class Config(BaseModel):
    api: ApiConfig = Field(default_factory=ApiConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def load(cls) -> "Config":
        """Load configuration with precedence: CLI > env > project > global > defaults"""
        config_data = {}

        # 1. Load global config
        global_config_path = Path.home() / ".agentcore" / "config.yaml"
        if global_config_path.exists():
            with open(global_config_path) as f:
                config_data = yaml.safe_load(f) or {}

        # 2. Load project config (override global)
        project_config_path = Path.cwd() / ".agentcore.yaml"
        if project_config_path.exists():
            with open(project_config_path) as f:
                project_data = yaml.safe_load(f) or {}
                config_data = deep_merge(config_data, project_data)

        # 3. Load environment variables (override files)
        env_overrides = cls._load_from_env()
        config_data = deep_merge(config_data, env_overrides)

        return cls(**config_data)

    @staticmethod
    def _load_from_env() -> dict:
        """Load configuration from environment variables"""
        env_mapping = {
            "AGENTCORE_API_URL": ["api", "url"],
            "AGENTCORE_API_TIMEOUT": ["api", "timeout"],
            "AGENTCORE_TOKEN": ["auth", "token"],
            "AGENTCORE_OUTPUT_FORMAT": ["output", "format"],
        }

        result = {}
        for env_var, path in env_mapping.items():
            value = os.getenv(env_var)
            if value:
                set_nested(result, path, value)

        return result
```

### Week 2: Core Commands

#### Days 6-8: Agent Commands (5 SP)

**Tasks:**

- Implement `agent register` command
- Implement `agent list` command
- Implement `agent info` command
- Implement `agent remove` command
- Implement `agent search` command
- Add input validation
- Add output formatting

**Implementation:**

```python
# src/agentcore_cli/commands/agent.py
import typer
from typing import List, Optional
from agentcore_cli.client import AgentCoreClient
from agentcore_cli.config import Config
from agentcore_cli.formatters import format_table, format_json

agent_app = typer.Typer(help="Manage agents")

@agent_app.command()
def register(
    name: str = typer.Option(..., help="Agent name"),
    capabilities: str = typer.Option(..., help="Comma-separated capabilities"),
    cost_per_request: float = typer.Option(0.01, help="Cost per request"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Register a new agent"""
    config = Config.load()
    client = AgentCoreClient(str(config.api.url))

    result = client.call("agent.register", {
        "name": name,
        "capabilities": capabilities.split(","),
        "cost_per_request": cost_per_request
    })

    if json_output:
        typer.echo(format_json(result))
    else:
        typer.echo(f"✓ Agent registered: {result['agent_id']}")

@agent_app.command()
def list(
    status: Optional[str] = typer.Option(None, help="Filter by status"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """List all agents"""
    config = Config.load()
    client = AgentCoreClient(str(config.api.url))

    params = {}
    if status:
        params["status"] = status

    result = client.call("agent.list", params)

    if json_output:
        typer.echo(format_json(result))
    else:
        typer.echo(format_table(result["agents"], columns=["agent_id", "name", "status", "capabilities"]))
```

**Testing:**

```python
# tests/test_agent_commands.py
from typer.testing import CliRunner
from agentcore_cli.cli import app

runner = CliRunner()

def test_agent_register():
    result = runner.invoke(app, [
        "agent", "register",
        "--name", "test-agent",
        "--capabilities", "python,testing"
    ])
    assert result.exit_code == 0
    assert "Agent registered" in result.stdout
```

---

## Sprint 2: Advanced Features (Week 3-4, 16 SP)

### Week 3: Session and Workflow Commands

#### Days 9-11: Task Commands (5 SP)

**Tasks:**

- Implement `task create` command
- Implement `task status` command with `--watch` flag
- Implement `task list` command with filtering
- Implement `task cancel` command
- Implement `task result` command
- Add progress indicators for long operations

**Watch Mode Implementation:**

```python
@task_app.command()
def status(
    task_id: str,
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch status updates"),
    interval: int = typer.Option(5, help="Watch interval in seconds"),
):
    """Get task status"""
    config = Config.load()
    client = AgentCoreClient(str(config.api.url))

    if watch:
        import time
        from rich.live import Live
        from rich.table import Table

        with Live(refresh_per_second=1) as live:
            while True:
                result = client.call("task.status", {"task_id": task_id})
                table = create_status_table(result)
                live.update(table)

                if result["status"] in ["completed", "failed"]:
                    break

                time.sleep(interval)
    else:
        result = client.call("task.status", {"task_id": task_id})
        typer.echo(format_table([result]))
```

#### Days 12-13: Session Commands (5 SP)

**Tasks:**

- Implement `session save` command
- Implement `session resume` command
- Implement `session list` command
- Implement `session info` command
- Implement `session delete` command
- Add session metadata support (tags, description)

**Implementation:**

```python
# src/agentcore_cli/commands/session.py
session_app = typer.Typer(help="Manage workflow sessions")

@session_app.command()
def save(
    name: str = typer.Option(..., help="Session name"),
    description: str = typer.Option("", help="Session description"),
    tags: List[str] = typer.Option([], help="Session tags"),
):
    """Save current workflow session"""
    config = Config.load()
    client = AgentCoreClient(str(config.api.url))

    result = client.call("session.save", {
        "name": name,
        "description": description,
        "tags": tags,
        "metadata": {}
    })

    typer.echo(f"✓ Session saved: {result['session_id']}")
    typer.echo(f"  Resume with: agentcore session resume {result['session_id']}")

@session_app.command()
def resume(session_id: str):
    """Resume a saved session"""
    config = Config.load()
    client = AgentCoreClient(str(config.api.url))

    with typer.progressbar(length=100, label="Restoring session") as progress:
        result = client.call("session.resume", {"session_id": session_id})
        progress.update(100)

    typer.echo(f"✓ Session resumed: {result['session_id']}")
    typer.echo(f"  Tasks restored: {result['tasks_count']}")
    typer.echo(f"  Agents restored: {result['agents_count']}")
```

#### Days 14: Workflow Commands (3 SP)

**Tasks:**

- Implement `workflow create` command
- Implement `workflow execute` command
- Implement `workflow status` command
- Implement `workflow list` command
- Support workflow definition files (YAML)

### Week 4: Polish and Testing

#### Days 15-16: Output Formatters (3 SP)

**Tasks:**

- Implement JSON formatter
- Implement table formatter (using rich)
- Implement tree formatter
- Add color support
- Add timestamp options

**Rich Table Implementation:**

```python
# src/agentcore_cli/formatters.py
from rich.console import Console
from rich.table import Table
from typing import List, Dict, Any

def format_table(
    data: List[Dict[str, Any]],
    columns: List[str] = None,
    title: str = None
) -> str:
    """Format data as a rich table"""
    console = Console()
    table = Table(title=title, show_header=True, header_style="bold magenta")

    if not data:
        return "No data"

    # Auto-detect columns if not specified
    if not columns:
        columns = list(data[0].keys())

    # Add columns
    for col in columns:
        table.add_column(col.replace("_", " ").title())

    # Add rows
    for row in data:
        table.add_row(*[str(row.get(col, "")) for col in columns])

    # Capture output
    with console.capture() as capture:
        console.print(table)

    return capture.get()
```

#### Days 17-18: Interactive Features (3 SP)

**Tasks:**

- Add confirmation prompts for destructive operations
- Add interactive wizards (optional)
- Implement progress bars
- Add `--watch` support for long operations

#### Days 19-20: Testing and Documentation (2 SP)

**Tasks:**

- Write unit tests (target: 90%+ coverage)
- Write integration tests with mock API
- Write E2E tests (optional, Docker Compose)
- Update README with installation and usage
- Generate CLI reference documentation

**Test Coverage Strategy:**

```bash
# Run tests with coverage
pytest --cov=agentcore_cli --cov-report=html --cov-report=term

# Target: 90%+ coverage
# Key areas:
# - Command parsing: 100%
# - JSON-RPC client: 95%
# - Configuration: 95%
# - Formatters: 90%
# - Error handling: 95%
```

---

## Technical Architecture

### Package Structure

```
agentcore-cli/
├── pyproject.toml          # Poetry/setuptools config
├── README.md               # Installation and quick start
├── CHANGELOG.md            # Version history
├── LICENSE                 # MIT License
├── .github/
│   └── workflows/
│       ├── test.yml        # CI/CD pipeline
│       └── publish.yml     # PyPI publishing
├── src/
│   └── agentcore_cli/
│       ├── __init__.py     # Package version
│       ├── cli.py          # Main CLI entry point
│       ├── client.py       # JSON-RPC client
│       ├── config.py       # Configuration management
│       ├── exceptions.py   # Custom exceptions
│       ├── formatters.py   # Output formatters
│       ├── utils.py        # Helper functions
│       └── commands/
│           ├── __init__.py
│           ├── agent.py    # Agent commands
│           ├── task.py     # Task commands
│           ├── session.py  # Session commands
│           ├── workflow.py # Workflow commands
│           └── config.py   # Config commands
├── tests/
│   ├── __init__.py
│   ├── conftest.py         # Pytest fixtures
│   ├── test_cli.py         # CLI tests
│   ├── test_client.py      # Client tests
│   ├── test_config.py      # Config tests
│   ├── test_formatters.py  # Formatter tests
│   └── integration/
│       └── test_e2e.py     # End-to-end tests
└── docs/
    ├── installation.md     # Installation guide
    ├── commands.md         # Command reference
    └── configuration.md    # Configuration guide
```

### Dependencies

**pyproject.toml:**

```toml
[project]
name = "agentcore-cli"
version = "0.1.0"
description = "Command-line interface for AgentCore"
requires-python = ">=3.12"
dependencies = [
    "typer[all]>=0.9.0",
    "requests>=2.31.0",
    "pyyaml>=6.0",
    "pydantic>=2.0.0",
    "rich>=13.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.11.0",
    "black>=23.7.0",
    "mypy>=1.5.0",
    "ruff>=0.0.287",
]

[project.scripts]
agentcore = "agentcore_cli.cli:main"

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"
```

---

## Testing Strategy

### Unit Tests (90%+ Coverage Target)

**Test Pyramid:**

- Unit tests: 70% of test suite
- Integration tests: 20% of test suite
- E2E tests: 10% of test suite

**Coverage by Module:**

| Module | Target Coverage | Priority |
|--------|-----------------|----------|
| cli.py | 100% | High |
| client.py | 95% | High |
| config.py | 95% | High |
| commands/*.py | 90% | High |
| formatters.py | 90% | Medium |
| utils.py | 85% | Medium |

### Integration Tests

**Mock API Server:**

```python
# tests/conftest.py
import pytest
from unittest.mock import Mock

@pytest.fixture
def mock_api_server():
    """Mock AgentCore API server"""
    def _handler(method: str, params: dict):
        if method == "agent.list":
            return {"agents": []}
        elif method == "agent.register":
            return {"agent_id": "agent-12345"}
        # Add more handlers

    return _handler
```

### E2E Tests (Optional)

**Docker Compose Test Environment:**

```yaml
# tests/docker-compose.test.yml
version: '3.8'
services:
  agentcore:
    image: agentcore:latest
    ports:
      - "8001:8001"

  cli-test:
    build: .
    depends_on:
      - agentcore
    environment:
      - AGENTCORE_API_URL=http://agentcore:8001
    command: pytest tests/integration/test_e2e.py
```

---

## Distribution and Deployment

### PyPI Publishing

**Build and Publish:**

```bash
# Build distribution
python -m build

# Test with TestPyPI
python -m twine upload --repository testpypi dist/*

# Publish to PyPI
python -m twine upload dist/*
```

**GitHub Actions Workflow:**

```yaml
# .github/workflows/publish.yml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Build package
        run: |
          pip install build
          python -m build
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

### Installation Methods

**PyPI (production):**

```bash
pip install agentcore-cli
```

**UV (recommended for AgentCore users):**

```bash
uv add agentcore-cli
```

**Development:**

```bash
git clone https://github.com/agentcore/agentcore-cli
cd agentcore-cli
pip install -e ".[dev]"
```

---

## Success Criteria

### Functional Requirements

- ✅ All agent commands working (register, list, info, remove, search)
- ✅ All task commands working (create, status, list, cancel, result)
- ✅ Session management commands (save, resume, list, info, delete)
- ✅ Workflow commands (create, execute, status, list)
- ✅ Configuration management (init, show, validate)
- ✅ Multiple output formats (JSON, table, tree)

### Non-Functional Requirements

- ✅ 90%+ code coverage
- ✅ <200ms startup time for simple commands
- ✅ <50MB memory usage
- ✅ User-friendly error messages
- ✅ Comprehensive documentation

### User Experience

- ✅ Time-to-first-task < 5 minutes (vs hours for API)
- ✅ Intuitive command structure (docker/kubectl familiar)
- ✅ Helpful error messages with suggestions
- ✅ Interactive prompts for complex operations

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| CLI framework choice (Click vs Typer) | Low | Medium | Evaluate both in first 2 days, choose based on type safety and DX |
| Session API not ready | Medium | High | Mock session commands, integrate when API available |
| Performance issues | Low | Medium | Profile early, optimize hot paths, lazy import heavy libraries |
| Testing complexity | Medium | Medium | Start with unit tests, add integration tests incrementally |

---

## Future Enhancements (Post-MVP)

### Phase 2 Features

1. Shell completion (bash, zsh, fish)
2. Interactive REPL mode
3. Batch operations from file
4. Advanced filtering (JMESPath, SQL-like)
5. Workflow templates and marketplace

### Phase 3 Features

1. Plugin architecture for custom commands
2. Custom output formatters
3. Secure credential storage (keychain)
4. TUI (Terminal UI) mode
5. Remote config management

---

## Timeline Summary

| Week | Sprint | Focus | Deliverables | Story Points |
|------|--------|-------|--------------|--------------|
| 1 | Sprint 1 | Foundation | CLI framework, JSON-RPC client, config | 8 SP |
| 2 | Sprint 1 | Core Commands | Agent commands, basic testing | 10 SP |
| 3 | Sprint 2 | Advanced Commands | Task, session, workflow commands | 13 SP |
| 4 | Sprint 2 | Polish | Formatters, testing, documentation | 3 SP |

**Total:** 4 weeks, 34 story points

---

## References

- [CLI Layer Specification](spec.md)
- [CLI Layer Tasks](tasks.md)
- [A2A Protocol API Reference](../a2a-protocol/spec.md)
- [Typer Documentation](https://typer.tiangolo.com/)
- [Rich Documentation](https://rich.readthedocs.io/)

---

**End of Implementation Plan**
