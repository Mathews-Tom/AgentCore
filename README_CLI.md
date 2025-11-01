# AgentCore CLI

Developer-friendly command-line interface for AgentCore, providing easy access to the A2A protocol JSON-RPC 2.0 API.

**Version:** 2.0 (Redesigned Architecture)
**Python:** 3.12+
**Framework:** Typer + Rich
**Protocol:** JSON-RPC 2.0 compliant

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Commands](#commands)
  - [Agent Commands](#agent-commands)
  - [Task Commands](#task-commands)
  - [Session Commands](#session-commands)
  - [Workflow Commands](#workflow-commands)
  - [Config Commands](#config-commands)
- [Output Formats](#output-formats)
- [Authentication](#authentication)
- [Error Handling](#error-handling)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

---

## Installation

### From Source

```bash
# Clone repository
git clone https://github.com/your-org/agentcore.git
cd agentcore

# Install dependencies using uv
uv sync

# Run CLI
uv run agentcore --help
```

### Using pip (future)

```bash
pip install agentcore-cli
agentcore --help
```

---

## Quick Start

### 1. Start the API Server

```bash
# Using Docker Compose (recommended)
docker compose -f docker-compose.dev.yml up

# Or run directly
uv run uvicorn agentcore.a2a_protocol.main:app --host 0.0.0.0 --port 8001
```

### 2. Configure CLI

```bash
# Initialize configuration
agentcore config init

# Set API URL
agentcore config set api.url http://localhost:8001

# Verify connection
agentcore health
```

### 3. Register an Agent

```bash
# Register agent
agentcore agent register \
  --name analyzer \
  --capabilities "python,analysis,testing"

# List agents
agentcore agent list

# Get agent details
agentcore agent info agent-001
```

### 4. Create and Execute Tasks

```bash
# Create task
agentcore task create \
  --description "Analyze Python code for bugs" \
  --agent-id agent-001

# Monitor task
agentcore task info task-001

# View task logs
agentcore task logs task-001 --follow
```

---

## Architecture

The CLI follows a strict 4-layer architecture for maintainability, testability, and A2A protocol compliance:

```plaintext
┌─────────────────────────────────────────┐
│  Layer 1: CLI Layer (Typer)             │
│  - Argument parsing & validation        │
│  - User interaction & prompts           │
│  - Output formatting (table/JSON)       │
│  - Exit code handling                   │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Layer 2: Service Layer (Facade)        │
│  - Business operations                  │
│  - Parameter validation                 │
│  - Domain error handling                │
│  - NO JSON-RPC knowledge                │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Layer 3: Protocol Layer (JSON-RPC)     │
│  - JSON-RPC 2.0 specification           │
│  - Request/Response models (Pydantic)   │
│  - A2A context management               │
│  - Protocol error translation           │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Layer 4: Transport Layer (HTTP)        │
│  - HTTP communication                   │
│  - Connection pooling                   │
│  - Retry logic & backoff                │
│  - SSL/TLS & timeout handling           │
└─────────────────────────────────────────┘
                 │
                 ▼
         AgentCore API Server
         (JSON-RPC 2.0 Endpoint)
```

### Design Principles

1. **Clear Separation of Concerns**: Each layer has a single, well-defined responsibility
2. **A2A Protocol Compliance**: JSON-RPC 2.0 with proper `params` wrapper
3. **Testability**: Mock at service layer for fast, isolated CLI tests
4. **Type Safety**: Full type hints with Pydantic validation
5. **Extensibility**: Easy to add new commands and services

---

## Configuration

### Configuration Precedence

Configuration is loaded in the following order (highest to lowest priority):

1. **CLI arguments** (e.g., `--api-url`)
2. **Environment variables** (e.g., `AGENTCORE_API_URL`)
3. **Project config** (`.agentcore.toml` in current directory)
4. **Global config** (`~/.agentcore/config.toml`)
5. **Defaults**

### Configuration File Schema

```toml
# .agentcore.toml or ~/.agentcore/config.toml

[api]
url = "http://localhost:8001"
timeout = 30        # Request timeout in seconds
retries = 3         # Number of retry attempts
verify_ssl = true   # SSL/TLS verification

[auth]
type = "jwt"        # Options: "none", "jwt", "api_key"
token = ""          # JWT token or API key

[defaults]
output_format = "table"  # Options: "table", "json"
limit = 100              # Default limit for list commands
```

### Environment Variables

All configuration can be set via environment variables with `AGENTCORE_` prefix:

```bash
export AGENTCORE_API_URL="http://localhost:8001"
export AGENTCORE_API_TIMEOUT=30
export AGENTCORE_AUTH_TYPE="jwt"
export AGENTCORE_AUTH_TOKEN="your-jwt-token"
```

### Managing Configuration

```bash
# Initialize configuration (creates ~/.agentcore/config.toml)
agentcore config init

# Show current configuration
agentcore config show

# Show as JSON
agentcore config show --json

# Set configuration value
agentcore config set api.url http://prod.example.com
agentcore config set auth.type jwt
agentcore config set auth.token <your-token>

# Get specific value
agentcore config get api.url

# Use project-level config
agentcore config init --project
```

---

## Commands

### Agent Commands

#### `agent register`

Register a new agent with the system.

```bash
agentcore agent register --name NAME --capabilities CAPS [OPTIONS]

Options:
  --name, -n TEXT              Agent name (required)
  --capabilities, -c TEXT      Comma-separated capabilities (required)
  --cost-per-request FLOAT     Cost per request in dollars [default: 0.01]
  --json, -j                   Output in JSON format
  --help                       Show help message

Examples:
  # Basic registration
  agentcore agent register -n analyzer -c "python,analysis"

  # With custom cost
  agentcore agent register \
    --name tester \
    --capabilities "testing,qa,automation" \
    --cost-per-request 0.05

  # JSON output
  agentcore agent register -n coder -c "python,go" --json
```

#### `agent list`

List all registered agents.

```bash
agentcore agent list [OPTIONS]

Options:
  --status TEXT          Filter by status (active, inactive, error)
  --limit, -l INTEGER    Maximum number of results [default: 100]
  --json, -j             Output in JSON format
  --help                 Show help message

Examples:
  # List all agents
  agentcore agent list

  # Filter by status
  agentcore agent list --status active

  # Limit results
  agentcore agent list --limit 10

  # JSON output
  agentcore agent list --json
```

#### `agent info`

Get detailed information about a specific agent.

```bash
agentcore agent info AGENT_ID [OPTIONS]

Arguments:
  AGENT_ID               Agent ID (required)

Options:
  --json, -j             Output in JSON format
  --help                 Show help message

Examples:
  # Get agent info
  agentcore agent info agent-001

  # JSON output
  agentcore agent info agent-001 --json
```

#### `agent remove`

Remove an agent from the system.

```bash
agentcore agent remove AGENT_ID [OPTIONS]

Arguments:
  AGENT_ID               Agent ID (required)

Options:
  --force, -f            Skip confirmation prompt
  --json, -j             Output in JSON format
  --help                 Show help message

Examples:
  # Remove agent (with confirmation)
  agentcore agent remove agent-001

  # Force removal
  agentcore agent remove agent-001 --force

  # JSON output
  agentcore agent remove agent-001 -f --json
```

#### `agent search`

Search for agents by capability.

```bash
agentcore agent search --capability CAP [OPTIONS]

Options:
  --capability, -c TEXT  Capability to search for (required)
  --limit, -l INTEGER    Maximum number of results [default: 100]
  --json, -j             Output in JSON format
  --help                 Show help message

Examples:
  # Search by capability
  agentcore agent search -c python

  # Limit results
  agentcore agent search --capability analysis --limit 5

  # JSON output
  agentcore agent search -c testing --json
```

---

### Task Commands

#### `task create`

Create a new task.

```bash
agentcore task create --description DESC [OPTIONS]

Options:
  --description, -d TEXT       Task description (required)
  --agent-id TEXT              Assign to specific agent
  --priority INTEGER           Priority (1-5) [default: 3]
  --timeout INTEGER            Timeout in seconds
  --json, -j                   Output in JSON format
  --help                       Show help message

Examples:
  # Create task
  agentcore task create -d "Analyze code quality"

  # Assign to specific agent
  agentcore task create \
    --description "Run tests" \
    --agent-id agent-001

  # With priority and timeout
  agentcore task create \
    -d "Critical bug fix" \
    --priority 5 \
    --timeout 300

  # JSON output
  agentcore task create -d "Deploy service" --json
```

#### `task list`

List all tasks.

```bash
agentcore task list [OPTIONS]

Options:
  --status TEXT          Filter by status (pending, running, completed, failed)
  --agent-id TEXT        Filter by agent
  --limit, -l INTEGER    Maximum number of results [default: 100]
  --json, -j             Output in JSON format
  --help                 Show help message

Examples:
  # List all tasks
  agentcore task list

  # Filter by status
  agentcore task list --status running

  # Filter by agent
  agentcore task list --agent-id agent-001

  # JSON output
  agentcore task list --json
```

#### `task info`

Get detailed information about a task.

```bash
agentcore task info TASK_ID [OPTIONS]

Arguments:
  TASK_ID                Task ID (required)

Options:
  --json, -j             Output in JSON format
  --help                 Show help message

Examples:
  # Get task info
  agentcore task info task-001

  # JSON output
  agentcore task info task-001 --json
```

#### `task cancel`

Cancel a running task.

```bash
agentcore task cancel TASK_ID [OPTIONS]

Arguments:
  TASK_ID                Task ID (required)

Options:
  --force, -f            Force cancellation
  --json, -j             Output in JSON format
  --help                 Show help message

Examples:
  # Cancel task
  agentcore task cancel task-001

  # Force cancel
  agentcore task cancel task-001 --force

  # JSON output
  agentcore task cancel task-001 --json
```

#### `task logs`

View task execution logs.

```bash
agentcore task logs TASK_ID [OPTIONS]

Arguments:
  TASK_ID                Task ID (required)

Options:
  --follow, -f           Follow log output
  --lines, -n INTEGER    Number of lines to show [default: 100]
  --help                 Show help message

Examples:
  # View logs
  agentcore task logs task-001

  # Follow logs (live)
  agentcore task logs task-001 --follow

  # Show last 50 lines
  agentcore task logs task-001 --lines 50
```

---

### Session Commands

#### `session create`

Create a new session.

```bash
agentcore session create --name NAME [OPTIONS]

Options:
  --name, -n TEXT        Session name (required)
  --description TEXT     Session description
  --json, -j             Output in JSON format
  --help                 Show help message

Examples:
  # Create session
  agentcore session create -n "dev-session"

  # With description
  agentcore session create \
    --name "prod-deployment" \
    --description "Production deployment session"

  # JSON output
  agentcore session create -n "test-session" --json
```

#### `session list`

List all sessions.

```bash
agentcore session list [OPTIONS]

Options:
  --state TEXT           Filter by state (active, paused, completed)
  --limit, -l INTEGER    Maximum number of results [default: 100]
  --json, -j             Output in JSON format
  --help                 Show help message

Examples:
  # List all sessions
  agentcore session list

  # Filter by state
  agentcore session list --state active

  # JSON output
  agentcore session list --json
```

#### `session info`

Get detailed information about a session.

```bash
agentcore session info SESSION_ID [OPTIONS]

Arguments:
  SESSION_ID             Session ID (required)

Options:
  --json, -j             Output in JSON format
  --help                 Show help message

Examples:
  # Get session info
  agentcore session info session-001

  # JSON output
  agentcore session info session-001 --json
```

#### `session delete`

Delete a session.

```bash
agentcore session delete SESSION_ID [OPTIONS]

Arguments:
  SESSION_ID             Session ID (required)

Options:
  --force, -f            Skip confirmation prompt
  --json, -j             Output in JSON format
  --help                 Show help message

Examples:
  # Delete session
  agentcore session delete session-001

  # Force delete
  agentcore session delete session-001 --force

  # JSON output
  agentcore session delete session-001 -f --json
```

---

### Workflow Commands

#### `workflow run`

Execute a workflow from a YAML file.

```bash
agentcore workflow run --file FILE [OPTIONS]

Options:
  --file, -f FILE        Workflow YAML file (required)
  --json, -j             Output in JSON format
  --help                 Show help message

Examples:
  # Run workflow
  agentcore workflow run -f deploy.yaml

  # JSON output
  agentcore workflow run --file ci-cd.yaml --json
```

#### `workflow list`

List all workflows.

```bash
agentcore workflow list [OPTIONS]

Options:
  --status TEXT          Filter by status (pending, running, completed, failed)
  --limit, -l INTEGER    Maximum number of results [default: 100]
  --json, -j             Output in JSON format
  --help                 Show help message

Examples:
  # List all workflows
  agentcore workflow list

  # Filter by status
  agentcore workflow list --status running

  # JSON output
  agentcore workflow list --json
```

#### `workflow info`

Get detailed information about a workflow.

```bash
agentcore workflow info WORKFLOW_ID [OPTIONS]

Arguments:
  WORKFLOW_ID            Workflow ID (required)

Options:
  --json, -j             Output in JSON format
  --help                 Show help message

Examples:
  # Get workflow info
  agentcore workflow info workflow-001

  # JSON output
  agentcore workflow info workflow-001 --json
```

#### `workflow stop`

Stop a running workflow.

```bash
agentcore workflow stop WORKFLOW_ID [OPTIONS]

Arguments:
  WORKFLOW_ID            Workflow ID (required)

Options:
  --force, -f            Force stop
  --json, -j             Output in JSON format
  --help                 Show help message

Examples:
  # Stop workflow
  agentcore workflow stop workflow-001

  # Force stop
  agentcore workflow stop workflow-001 --force

  # JSON output
  agentcore workflow stop workflow-001 -f --json
```

---

### Config Commands

#### `config show`

Display current configuration.

```bash
agentcore config show [OPTIONS]

Options:
  --global               Show global config (~/.agentcore/config.toml)
  --project              Show project config (.agentcore.toml)
  --json, -j             Output in JSON format
  --help                 Show help message

Examples:
  # Show merged config
  agentcore config show

  # Show global config only
  agentcore config show --global

  # Show as JSON
  agentcore config show --json
```

#### `config set`

Set configuration value.

```bash
agentcore config set KEY VALUE [OPTIONS]

Arguments:
  KEY                    Configuration key (e.g., api.url)
  VALUE                  Configuration value

Options:
  --global               Set in global config
  --project              Set in project config
  --help                 Show help message

Examples:
  # Set API URL
  agentcore config set api.url http://localhost:8001

  # Set auth token (global)
  agentcore config set auth.token <token> --global

  # Set timeout (project)
  agentcore config set api.timeout 60 --project
```

#### `config get`

Get configuration value.

```bash
agentcore config get KEY [OPTIONS]

Arguments:
  KEY                    Configuration key (e.g., api.url)

Options:
  --global               Get from global config
  --project              Get from project config
  --help                 Show help message

Examples:
  # Get API URL
  agentcore config get api.url

  # Get auth token from global config
  agentcore config get auth.token --global
```

#### `config init`

Initialize configuration file.

```bash
agentcore config init [OPTIONS]

Options:
  --global               Initialize global config (~/.agentcore/config.toml)
  --project              Initialize project config (.agentcore.toml)
  --help                 Show help message

Examples:
  # Initialize global config
  agentcore config init --global

  # Initialize project config
  agentcore config init --project
```

---

## Output Formats

The CLI supports two output formats: **table** (default) and **JSON**.

### Table Format (Default)

Human-readable table output with rich formatting:

```bash
$ agentcore agent list

┌────────────┬───────────┬────────┬──────────────────────┐
│ Agent ID   │ Name      │ Status │ Capabilities         │
├────────────┼───────────┼────────┼──────────────────────┤
│ agent-001  │ analyzer  │ active │ python, analysis     │
│ agent-002  │ tester    │ active │ testing, qa          │
│ agent-003  │ coder     │ active │ python, go, rust     │
└────────────┴───────────┴────────┴──────────────────────┘
```

### JSON Format

Machine-readable JSON output (use `--json` or `-j` flag):

```bash
$ agentcore agent list --json

[
  {
    "agent_id": "agent-001",
    "name": "analyzer",
    "status": "active",
    "capabilities": ["python", "analysis"],
    "cost_per_request": 0.01,
    "created_at": "2025-10-22T10:00:00Z"
  },
  {
    "agent_id": "agent-002",
    "name": "tester",
    "status": "active",
    "capabilities": ["testing", "qa"],
    "cost_per_request": 0.01,
    "created_at": "2025-10-22T10:05:00Z"
  }
]
```

JSON output is ideal for:
- Scripting and automation
- Parsing with `jq` or other tools
- Integration with CI/CD pipelines
- Programmatic access

---

## Authentication

The CLI supports multiple authentication methods:

### No Authentication

```bash
agentcore config set auth.type none
```

### JWT Authentication

```bash
# Set auth type
agentcore config set auth.type jwt

# Set token
agentcore config set auth.token <your-jwt-token>

# Or use environment variable
export AGENTCORE_AUTH_TOKEN="your-jwt-token"
```

### API Key Authentication

```bash
# Set auth type
agentcore config set auth.type api_key

# Set API key
agentcore config set auth.token <your-api-key>

# Or use environment variable
export AGENTCORE_AUTH_TOKEN="your-api-key"
```

---

## Error Handling

The CLI uses standard Unix exit codes:

| Exit Code | Meaning | Example |
|-----------|---------|---------|
| 0 | Success | Command completed successfully |
| 1 | General error | Agent not found, operation failed |
| 2 | Usage error | Invalid arguments, validation failed |
| 3 | Connection error | Cannot reach API server |
| 4 | Authentication error | Invalid or expired token |

### Error Messages

Errors are displayed with clear formatting:

```bash
$ agentcore agent info invalid-id

Error: Agent not found
Agent 'invalid-id' does not exist in the system

$ echo $?
1
```

### Validation Errors

```bash
$ agentcore agent register --name "" --capabilities "python"

Validation error: Agent name cannot be empty

$ echo $?
2
```

### Connection Errors

```bash
$ agentcore agent list

Connection error: Cannot reach API server at http://localhost:8001
Please verify:
  - API server is running
  - URL is correct: agentcore config get api.url
  - Network connectivity

$ echo $?
3
```

---

## Troubleshooting

### Issue: Cannot connect to API server

**Error:**
```
Connection error: Cannot reach API server at http://localhost:8001
```

**Solution:**
1. Verify API server is running:
   ```bash
   docker compose -f docker-compose.dev.yml ps
   ```
2. Check API URL configuration:
   ```bash
   agentcore config get api.url
   ```
3. Test API directly:
   ```bash
   curl http://localhost:8001/health
   ```
4. Update URL if needed:
   ```bash
   agentcore config set api.url http://localhost:8001
   ```

### Issue: Authentication failed

**Error:**
```
Authentication error: Invalid or expired token
```

**Solution:**
1. Check auth type:
   ```bash
   agentcore config get auth.type
   ```
2. Verify token:
   ```bash
   agentcore config get auth.token
   ```
3. Update token:
   ```bash
   agentcore config set auth.token <new-token>
   ```

### Issue: SSL certificate verification failed

**Error:**
```
SSL error: Certificate verification failed
```

**Solution (for development only):**
```bash
# Disable SSL verification (NOT for production)
agentcore config set api.verify_ssl false
```

### Issue: Command not found

**Error:**
```
agentcore: command not found
```

**Solution:**
```bash
# If running from source
uv run agentcore [COMMAND]

# Or add to PATH (future pip install)
pip install agentcore-cli
```

### Issue: Timeout errors

**Error:**
```
Timeout error: Request timed out after 30 seconds
```

**Solution:**
```bash
# Increase timeout
agentcore config set api.timeout 60
```

### Debug Mode

Enable verbose output for troubleshooting:

```bash
# Set log level to debug
export AGENTCORE_LOG_LEVEL=debug

# Run command
agentcore agent list
```

---

## Development

### Running from Source

```bash
# Install development dependencies
uv sync --dev

# Run CLI
uv run agentcore [COMMAND]

# Run with Python module
uv run python -m agentcore_cli [COMMAND]
```

### Running Tests

```bash
# Run all CLI tests
uv run pytest tests/cli/

# Run specific test file
uv run pytest tests/cli/test_agent_commands.py

# Run with coverage
uv run pytest tests/cli/ --cov=agentcore_cli --cov-report=html

# Run single test
uv run pytest tests/cli/test_agent_commands.py::test_register_success
```

### Type Checking

```bash
# Check types with mypy (strict mode)
uv run mypy src/agentcore_cli/

# Check specific file
uv run mypy src/agentcore_cli/commands/agent.py
```

### Linting

```bash
# Run Ruff linter
uv run ruff check src/agentcore_cli/

# Auto-fix issues
uv run ruff check --fix src/agentcore_cli/
```

### Adding New Commands

See the comprehensive migration guide for patterns and best practices:

- [CLI Migration Learnings](docs/architecture/cli-migration-learnings.md)
- [CLI Testing Guide](docs/architecture/cli-testing-guide.md)
- [CLI Specification](docs/specs/cli-layer/spec.md)

---

## Additional Resources

- **Architecture Documentation:** [docs/specs/cli-layer/spec.md](docs/specs/cli-layer/spec.md)
- **Migration Guide:** [docs/architecture/cli-migration-learnings.md](docs/architecture/cli-migration-learnings.md)
- **Testing Guide:** [docs/architecture/cli-testing-guide.md](docs/architecture/cli-testing-guide.md)
- **Validation Report:** [docs/architecture/cli-validation-report.md](docs/architecture/cli-validation-report.md)
- **A2A Protocol Specification:** [docs/specs/a2a-protocol/spec.md](docs/specs/a2a-protocol/spec.md)

---

## Support

For issues, questions, or contributions:

- **GitHub Issues:** https://github.com/your-org/agentcore/issues
- **Documentation:** https://agentcore.docs.example.com
- **Community:** https://discord.gg/agentcore

---

**License:** MIT
**Maintainers:** AgentCore Team
**Last Updated:** 2025-10-22
