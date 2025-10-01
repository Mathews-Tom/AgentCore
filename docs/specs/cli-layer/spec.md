# CLI Layer Specification

**Component:** CLI Layer
**Purpose:** Developer-friendly command-line interface for AgentCore
**Owner:** Platform Team
**Dependencies:** A2A Protocol Layer (JSON-RPC API)
**Created:** 2025-09-30
**Status:** Planned

---

## Overview

The CLI Layer provides a Python-based command-line interface that wraps the AgentCore JSON-RPC 2.0 API with developer-friendly commands. It lowers the barrier to entry for developers while maintaining full access to AgentCore's enterprise capabilities.

**Target Users:**

- Developers prototyping agent systems
- DevOps engineers managing agent deployments
- QA teams testing agent workflows
- Technical users preferring CLI over API integration

**Design Philosophy:**

- Mirror familiar CLI patterns (docker, kubectl, git)
- Sensible defaults with explicit overrides
- Progressive disclosure (simple → advanced)
- Machine-readable output for scripting

---

## Architecture

### Technology Stack

**Core Framework:**

- **Language:** Python 3.12+
- **CLI Framework:** Click or Typer (decision pending)
- **JSON-RPC Client:** requests library with retry logic
- **Output Formatting:** rich library for tables and trees
- **Configuration:** PyYAML for .agentcore.yaml files
- **Validation:** Pydantic for input/output validation

**Distribution:**

- **Package Name:** `agentcore-cli`
- **Entry Point:** `agentcore` command
- **Installation:** `pip install agentcore-cli` or `uv add agentcore-cli`
- **Deployment:** PyPI distribution

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CLI Layer                             │
│  ┌──────────────────────────────────────────────────┐   │
│  │           Command Parser (Click/Typer)           │   │
│  └────────────────┬─────────────────────────────────┘   │
│                   │                                      │
│  ┌────────────────▼─────────────────────────────────┐   │
│  │         Configuration Manager                     │   │
│  │  • Global config (~/.agentcore/config.yaml)      │   │
│  │  • Project config (./.agentcore.yaml)            │   │
│  │  • Environment variables (AGENTCORE_*)           │   │
│  └────────────────┬─────────────────────────────────┘   │
│                   │                                      │
│  ┌────────────────▼─────────────────────────────────┐   │
│  │         JSON-RPC Client                           │   │
│  │  • Request builder                                │   │
│  │  • Connection management                          │   │
│  │  • Retry logic with exponential backoff           │   │
│  │  • Error handling and translation                 │   │
│  └────────────────┬─────────────────────────────────┘   │
│                   │                                      │
│  ┌────────────────▼─────────────────────────────────┐   │
│  │         Output Formatter                          │   │
│  │  • JSON format (--json)                           │   │
│  │  • Table format (default)                         │   │
│  │  • Tree format (--tree)                           │   │
│  │  • Color support                                  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                       │
                       │ HTTP/JSON-RPC 2.0
                       ▼
┌─────────────────────────────────────────────────────────┐
│               AgentCore API Server                       │
│         (JSON-RPC 2.0 Endpoint)                          │
└─────────────────────────────────────────────────────────┘
```

---

## Command Structure

### Top-Level Commands

```bash
agentcore [OPTIONS] COMMAND [ARGS]...
```

**Global Options:**

- `--config PATH` - Config file path (default: ./.agentcore.yaml)
- `--api-url URL` - AgentCore API URL (default: <http://localhost:8001>)
- `--json` - Output JSON format
- `--verbose, -v` - Verbose output
- `--quiet, -q` - Suppress non-error output
- `--help, -h` - Show help message

### Agent Commands

**Purpose:** Manage agent lifecycle and discovery

```bash
# Register a new agent
agentcore agent register \
  --name "code-analyzer" \
  --capabilities "python,analysis,linting" \
  --requirements '{"memory": "512MB", "cpu": "0.5"}' \
  --cost-per-request 0.01

# List all agents
agentcore agent list [--filter STATUS] [--format json|table]

# Get agent details
agentcore agent info <agent-id>

# Update agent capabilities
agentcore agent update <agent-id> --capabilities "python,typescript,analysis"

# Remove agent
agentcore agent remove <agent-id> [--force]

# Search agents by capability
agentcore agent search --capability "python" --capability "testing"
```

### Task Commands

**Purpose:** Create, monitor, and manage tasks

```bash
# Create a new task
agentcore task create \
  --type "code-review" \
  --input "src/**/*.py" \
  --requirements '{"language": "python"}' \
  --priority high

# Get task status
agentcore task status <task-id> [--watch]

# List tasks
agentcore task list [--status pending|running|completed|failed] [--limit 10]

# Cancel task
agentcore task cancel <task-id>

# Get task result/artifacts
agentcore task result <task-id> [--output FILE]

# Retry failed task
agentcore task retry <task-id>
```

### Session Commands (NEW)

**Purpose:** Save and resume long-running workflows

```bash
# Save current session
agentcore session save \
  --name "feature-development" \
  --description "Building user authentication" \
  --tags "auth,backend"

# Resume session
agentcore session resume <session-id>

# List sessions
agentcore session list [--status active|paused|completed]

# Get session details
agentcore session info <session-id>

# Delete session
agentcore session delete <session-id>

# Export session for debugging
agentcore session export <session-id> --output session.json
```

### Workflow Commands

**Purpose:** Define and execute multi-step workflows

```bash
# Create workflow from definition file
agentcore workflow create --file workflow.yaml

# Execute workflow
agentcore workflow execute <workflow-id> [--watch]

# Get workflow status
agentcore workflow status <workflow-id>

# List workflows
agentcore workflow list

# Visualize workflow graph
agentcore workflow visualize <workflow-id> [--output graph.png]

# Pause/resume workflow
agentcore workflow pause <workflow-id>
agentcore workflow resume <workflow-id>
```

### Utility Commands

**Purpose:** Configuration, health checks, and system information

```bash
# Initialize config file
agentcore config init [--global]

# Show current configuration
agentcore config show

# Validate configuration
agentcore config validate

# Test API connectivity
agentcore health [--verbose]

# Show CLI version
agentcore version

# Show API server information
agentcore info
```

---

## Configuration Management

### Configuration Hierarchy

**Precedence Order (highest to lowest):**

1. CLI arguments (e.g., `--api-url`)
2. Environment variables (e.g., `AGENTCORE_API_URL`)
3. Project config (`./.agentcore.yaml`)
4. Global config (`~/.agentcore/config.yaml`)
5. Built-in defaults

### Configuration File Format

**Global Config:** `~/.agentcore/config.yaml`

```yaml
# AgentCore CLI Configuration

# API Connection
api:
  url: http://localhost:8001
  timeout: 30
  retries: 3
  verify_ssl: true

# Authentication (optional)
auth:
  type: jwt  # jwt | api_key | none
  token: ${AGENTCORE_TOKEN}
  # api_key: ${AGENTCORE_API_KEY}

# Output Preferences
output:
  format: table  # json | table | tree
  color: true
  timestamps: false
  verbose: false

# Defaults for commands
defaults:
  task:
    priority: medium
    timeout: 3600
  agent:
    cost_per_request: 0.01
```

**Project Config:** `./.agentcore.yaml`

```yaml
# Project-specific overrides

api:
  url: https://agentcore.example.com

defaults:
  task:
    requirements:
      language: python
      framework: fastapi
  workflow:
    max_retries: 3
    timeout: 7200
```

### Environment Variables

```bash
# API Configuration
export AGENTCORE_API_URL="http://localhost:8001"
export AGENTCORE_API_TIMEOUT="30"

# Authentication
export AGENTCORE_TOKEN="your-jwt-token"
export AGENTCORE_API_KEY="your-api-key"

# Output Preferences
export AGENTCORE_OUTPUT_FORMAT="json"
export AGENTCORE_COLOR="true"
export AGENTCORE_VERBOSE="false"

# SSL/TLS
export AGENTCORE_VERIFY_SSL="true"
export AGENTCORE_CA_BUNDLE="/path/to/ca-bundle.crt"
```

---

## Output Formats

### JSON Format (`--json`)

**Purpose:** Machine-readable output for scripting and automation

```json
{
  "jsonrpc": "2.0",
  "result": {
    "agent_id": "agent-12345",
    "name": "code-analyzer",
    "status": "active",
    "capabilities": ["python", "analysis"],
    "registered_at": "2025-09-30T10:30:00Z"
  },
  "id": 1
}
```

**Use Cases:**

- CI/CD pipeline integration
- Shell scripting
- Log aggregation
- API testing

### Table Format (default)

**Purpose:** Human-readable output with formatted columns

```
AGENT ID       NAME            STATUS    CAPABILITIES           REGISTERED
─────────────────────────────────────────────────────────────────────────────
agent-12345    code-analyzer   active    python, analysis       2025-09-30
agent-12346    test-runner     active    testing, python        2025-09-29
agent-12347    docs-generator  inactive  documentation, md      2025-09-28
```

**Use Cases:**

- Interactive CLI usage
- Quick status checks
- Dashboard-style output

### Tree Format (`--tree`)

**Purpose:** Hierarchical visualization for nested data

```
Workflow: feature-development
├── Task: setup-environment
│   ├── Status: completed
│   ├── Agent: deployment-agent
│   └── Duration: 45s
├── Task: run-tests
│   ├── Status: running
│   ├── Agent: test-runner
│   └── Progress: 67%
└── Task: deploy-application
    └── Status: pending
```

**Use Cases:**

- Workflow visualization
- Dependency graphs
- Session state inspection

---

## Error Handling

### Error Categories

**1. Connection Errors**

```
Error: Cannot connect to AgentCore API at http://localhost:8001
Suggestions:
  • Check if AgentCore server is running
  • Verify API URL in config: agentcore config show
  • Test connectivity: agentcore health
```

**2. Authentication Errors**

```
Error: Authentication failed (401 Unauthorized)
Suggestions:
  • Check your JWT token: echo $AGENTCORE_TOKEN
  • Regenerate token: agentcore auth login
  • Verify token expiration
```

**3. Validation Errors**

```
Error: Invalid task definition
  - Field 'requirements' is required
  - Field 'priority' must be one of: low, medium, high, critical

Use --help for command syntax
```

**4. API Errors**

```
Error: Task creation failed (JSON-RPC Error -32603)
Server message: "Agent with capability 'python' not found"
Suggestions:
  • List available agents: agentcore agent list
  • Register required agent: agentcore agent register --help
```

### Exit Codes

| Code | Meaning | Example |
|------|---------|---------|
| 0 | Success | Command completed successfully |
| 1 | General error | API returned error, command failed |
| 2 | Usage error | Invalid arguments, missing required fields |
| 3 | Connection error | Cannot connect to API server |
| 4 | Authentication error | Invalid or expired credentials |
| 5 | Timeout error | Operation exceeded timeout |
| 130 | Interrupted (Ctrl+C) | User cancelled operation |

---

## Interactive Features

### Progress Indicators

**Long-Running Operations:**

```bash
$ agentcore task create --type "large-analysis" --watch

Creating task... ✓
Assigning agent... ✓
Executing task...
  [████████████████░░░░░░░░] 67% (Processing file 45/67)
  Elapsed: 2m 15s | ETA: 1m 05s
```

**Async Operations with Follow:**

```bash
$ agentcore task status task-12345 --watch

Task: task-12345 | Status: running
Agent: code-analyzer | Started: 2m ago

Progress:
  [██████████████████░░░░░░] 75%

Press Ctrl+C to stop watching (task will continue)
```

### Interactive Prompts

**Confirmation Prompts:**

```bash
$ agentcore agent remove agent-12345

⚠️  Remove agent 'code-analyzer' (agent-12345)?
   This action cannot be undone.

   Continue? [y/N]:
```

**Interactive Wizards:**

```bash
$ agentcore workflow create --interactive

Workflow Creation Wizard
─────────────────────────────
? Workflow name: feature-deployment
? Description: Deploy new feature to production
? Add task 1/5: run-tests
  Agent capability: testing
? Add task 2/5: build-artifact
  Agent capability: build
? Add task 3/5: [Enter to skip]

✓ Workflow created: workflow-12345
  Review: agentcore workflow info workflow-12345
```

---

## Security Considerations

### Credential Management

**Never Store Plaintext Secrets:**

```yaml
# ❌ BAD - Hardcoded token
auth:
  token: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# ✅ GOOD - Environment variable reference
auth:
  token: ${AGENTCORE_TOKEN}
```

**Use System Keychain (Future):**

```bash
# Store token securely
agentcore auth login --save-keychain

# Token retrieved from OS keychain automatically
```

### TLS/SSL Verification

**Production (default):**

```yaml
api:
  url: https://agentcore.example.com
  verify_ssl: true
```

**Development (explicit opt-out):**

```bash
# Not recommended for production
agentcore --verify-ssl=false task list
```

### Input Validation

**Sanitize All Inputs:**

- Validate JSON schema before sending
- Escape special characters in strings
- Limit input size (prevent DOS)
- Validate file paths (prevent directory traversal)

---

## Testing Strategy

### Unit Tests (90%+ Coverage)

**Test Coverage:**

- Command parsing and validation
- Configuration loading and merging
- JSON-RPC client request building
- Output formatting (JSON, table, tree)
- Error handling and messages

**Framework:** pytest

```python
def test_agent_register_command():
    """Test agent register command with valid inputs"""
    result = runner.invoke(cli, [
        'agent', 'register',
        '--name', 'test-agent',
        '--capabilities', 'python,testing'
    ])
    assert result.exit_code == 0
    assert 'Agent registered' in result.output
```

### Integration Tests

**Test Scenarios:**

- End-to-end command execution against mock API
- Configuration precedence (CLI → env → file)
- Error handling for various API responses
- Output format correctness

**Mock JSON-RPC Responses:**

```python
@mock.patch('agentcore_cli.client.requests.post')
def test_task_create_integration(mock_post):
    mock_post.return_value = MockResponse({
        "jsonrpc": "2.0",
        "result": {"task_id": "task-123"},
        "id": 1
    })
    # Test command execution
```

### E2E Tests

**Test Against Real AgentCore:**

- Docker Compose test environment
- Full API interaction
- Multi-command workflows
- Session management flows

```bash
# E2E test script
agentcore agent register --name "e2e-agent" --capabilities "test"
task_id=$(agentcore task create --type "test" --json | jq -r '.result.task_id')
agentcore task status $task_id
agentcore agent remove $(agentcore agent list --json | jq -r '.result[0].agent_id')
```

---

## Performance Considerations

### Startup Time

**Target:** <200ms for simple commands (e.g., `agentcore --help`)

**Optimization Strategies:**

- Lazy import of heavy libraries (rich, pydantic)
- Minimal dependencies
- Compiled Python (.pyc) caching
- No unnecessary config loading for `--help`

### Network Optimization

**Connection Pooling:**

```python
# Reuse HTTP connection for multiple requests
session = requests.Session()
session.mount('http://', HTTPAdapter(pool_connections=1, pool_maxsize=1))
```

**Request Batching:**

```bash
# Batch multiple operations (future enhancement)
agentcore agent list | agentcore agent update --batch --status active
```

### Memory Usage

**Target:** <50MB for typical operations

**Strategies:**

- Stream large responses (don't load all in memory)
- Paginate list commands
- Clear caches after operations

---

## Extensibility

### Plugin Architecture (Future)

**Custom Commands:**

```python
# ~/.agentcore/plugins/custom_command.py
@agentcore_plugin.command()
def mycustom(ctx, arg):
    """Custom command description"""
    # Implementation
```

**Usage:**

```bash
agentcore mycustom --arg value
```

### Output Formatters (Future)

**Custom Format:**

```yaml
# config.yaml
output:
  custom_formatters:
    - name: csv
      module: agentcore_csv_formatter
```

---

## Migration Path

### From Direct API Usage

**Before (API):**

```bash
curl -X POST http://localhost:8001/api/v1/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"agent.register","params":{"name":"test"},"id":1}'
```

**After (CLI):**

```bash
agentcore agent register --name "test"
```

### From Other Tools

**Docker-like Familiarity:**

```bash
# Docker                      # AgentCore CLI
docker ps                   → agentcore agent list
docker inspect <id>         → agentcore agent info <id>
docker rm <id>              → agentcore agent remove <id>
docker run ...              → agentcore task create ...
```

**Kubectl-like Familiarity:**

```bash
# Kubectl                     # AgentCore CLI
kubectl get pods            → agentcore agent list
kubectl describe pod <id>   → agentcore agent info <id>
kubectl delete pod <id>     → agentcore agent remove <id>
kubectl apply -f ...        → agentcore workflow create --file ...
```

---

## Future Enhancements

### Phase 2 (Post-MVP)

1. **Shell Completion:**
   - Bash, Zsh, Fish completion scripts
   - Dynamic completion for IDs (agent-*, task-*)

2. **Interactive Mode:**
   - REPL-style interface
   - Command history and editing
   - Context-aware suggestions

3. **Batch Operations:**
   - Process multiple commands from file
   - Parallel execution support
   - Transaction-like semantics

4. **Advanced Filtering:**
   - JMESPath for JSON queries
   - SQL-like query language for list commands

5. **Workflow Templates:**
   - Pre-built workflow definitions
   - Template marketplace integration
   - Custom template authoring

---

## Dependencies

### Python Dependencies

**Required:**

- `click` or `typer` (CLI framework)
- `requests` (HTTP client)
- `pyyaml` (config file parsing)
- `pydantic` (validation)
- `rich` (output formatting)

**Optional:**

- `python-dotenv` (env file support)
- `keyring` (secure credential storage)
- `tabulate` (alternative table formatting)

### System Dependencies

**Runtime:**

- Python 3.12+ interpreter
- Internet connectivity to AgentCore API
- Terminal with color support (optional)

**Development:**

- pytest (testing)
- pytest-cov (coverage)
- black (formatting)
- mypy (type checking)
- ruff (linting)

---

## Success Metrics

### Adoption Metrics

- PyPI downloads per week
- GitHub stars on CLI repository
- Community contributions (PRs, issues)

### Usage Metrics

- CLI vs API usage ratio (target: 30%+ CLI)
- Command frequency distribution
- Error rate by command
- Average session duration

### Performance Metrics

- Command startup time (target: <200ms)
- API response time percentiles
- Memory usage (target: <50MB)
- Error handling effectiveness

---

## References

- [A2A Protocol Specification](../a2a-protocol/spec.md)
- [AgentCore API Documentation](../../README.md)
- [Click Documentation](https://click.palletsprojects.com/)
- [Typer Documentation](https://typer.tiangolo.com/)
- [Rich Documentation](https://rich.readthedocs.io/)

---

**End of Specification**
