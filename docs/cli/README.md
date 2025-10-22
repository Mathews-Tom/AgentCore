# AgentCore CLI

**Developer-friendly command-line interface for AgentCore**

The AgentCore CLI provides a simple, powerful way to interact with the AgentCore orchestration framework. Built on Google's A2A protocol, it offers intuitive commands for managing agents, tasks, workflows, and sessions.

---

## Features

- **Agent Management**: Register, discover, and manage agents with simple commands
- **Task Orchestration**: Create, monitor, and manage tasks with real-time status updates
- **Session Management**: Save and resume long-running workflows
- **Workflow Automation**: Define and execute multi-step workflows from YAML files
- **Multiple Output Formats**: JSON, table, and tree views for different use cases
- **Rich Formatting**: Beautiful terminal output with colors and progress indicators
- **Configuration Management**: Multi-level config (CLI → env → file → defaults)
- **Watch Mode**: Real-time monitoring of task and workflow execution

---

## Quick Start

### Installation

```bash
# Using pip
pip install agentcore

# Using uv (recommended)
uv add agentcore

# From source
git clone https://github.com/agentcore/agentcore.git
cd agentcore
pip install -e .
```

### Prerequisites

- Python 3.12 or higher
- AgentCore API server running (default: `http://localhost:8001`)

### First Steps

1. **Verify installation**:
   ```bash
   agentcore --version
   ```

2. **Check API connectivity**:
   ```bash
   agentcore config show
   ```

3. **Register your first agent**:
   ```bash
   agentcore agent register \
     --name "my-agent" \
     --capabilities "python,analysis"
   ```

4. **List registered agents**:
   ```bash
   agentcore agent list
   ```

5. **Create a task**:
   ```bash
   agentcore task create \
     --type "code-review" \
     --description "Review Python code for best practices"
   ```

---

## Command Overview

### Agent Commands

Manage agent lifecycle and discovery:

```bash
# Register a new agent
agentcore agent register \
  --name "code-analyzer" \
  --capabilities "python,analysis,linting"

# List all agents
agentcore agent list

# Get agent details
agentcore agent info <agent-id>

# Search agents by capability
agentcore agent search --capability "python"

# Remove an agent
agentcore agent remove <agent-id>
```

### Task Commands

Create, monitor, and manage tasks:

```bash
# Create a task
agentcore task create \
  --type "code-review" \
  --description "Review code quality"

# Check task status
agentcore task status <task-id>

# Watch task in real-time
agentcore task status <task-id> --watch

# List all tasks
agentcore task list

# Get task results
agentcore task result <task-id>

# Cancel a running task
agentcore task cancel <task-id>

# Retry a failed task
agentcore task retry <task-id>
```

### Session Commands

Save and resume long-running workflows:

```bash
# Save current session
agentcore session save \
  --name "feature-development" \
  --description "Building authentication feature"

# Resume a session
agentcore session resume <session-id>

# List sessions
agentcore session list

# Get session details
agentcore session info <session-id>

# Delete a session
agentcore session delete <session-id>

# Export session for debugging
agentcore session export <session-id> --output session.json
```

### Workflow Commands

Define and execute multi-step workflows:

```bash
# Create workflow from YAML
agentcore workflow create --file workflow.yaml

# Execute workflow
agentcore workflow execute <workflow-id>

# Watch workflow execution
agentcore workflow execute <workflow-id> --watch

# Get workflow status
agentcore workflow status <workflow-id>

# List workflows
agentcore workflow list

# Visualize workflow
agentcore workflow visualize <workflow-id>

# Pause/resume workflow
agentcore workflow pause <workflow-id>
agentcore workflow resume <workflow-id>
```

### Configuration Commands

Manage CLI configuration:

```bash
# Initialize config file
agentcore config init

# Show current configuration
agentcore config show

# Validate configuration
agentcore config validate
```

---

## Output Formats

### Table Format (Default)

Human-readable output with formatted columns:

```bash
agentcore agent list
```

```
                                Agents
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Agent Id     ┃ Name          ┃ Status ┃ Capabilities        ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ agent-12345  │ code-analyzer │ active │ python, analysis    │
│ agent-12346  │ test-runner   │ active │ testing, python     │
└──────────────┴───────────────┴────────┴─────────────────────┘

Total: 2 agent(s)
```

### JSON Format

Machine-readable output for scripting:

```bash
agentcore agent list --json
```

```json
{
  "jsonrpc": "2.0",
  "result": {
    "agents": [
      {
        "agent_id": "agent-12345",
        "name": "code-analyzer",
        "status": "active",
        "capabilities": ["python", "analysis"]
      }
    ]
  },
  "id": 1
}
```

### Tree Format

Hierarchical visualization:

```bash
agentcore workflow visualize <workflow-id>
```

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

---

## Configuration

### Configuration Hierarchy

AgentCore CLI uses a multi-level configuration system with the following precedence (highest to lowest):

1. **CLI arguments** (e.g., `--api-url http://localhost:8001`)
2. **Environment variables** (e.g., `AGENTCORE_API_URL`)
3. **Project config** (`./.agentcore.yaml`)
4. **Global config** (`~/.agentcore/config.yaml`)
5. **Built-in defaults**

### Configuration File

Create a configuration file with:

```bash
agentcore config init
```

This creates `~/.agentcore/config.yaml`:

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

### Environment Variables

```bash
# API Configuration
export AGENTCORE_API_URL="http://localhost:8001"
export AGENTCORE_API_TIMEOUT="30"

# Authentication
export AGENTCORE_TOKEN="your-jwt-token"

# Output Preferences
export AGENTCORE_OUTPUT_FORMAT="json"
export AGENTCORE_COLOR="true"
```

---

## Advanced Usage

### Watch Mode

Monitor long-running operations in real-time:

```bash
# Watch task execution
agentcore task status <task-id> --watch

# Watch workflow execution
agentcore workflow execute <workflow-id> --watch
```

Press `Ctrl+C` to stop watching (the task/workflow continues running).

### JSON Output for Scripting

Use JSON output for automation and scripting:

```bash
# Get task ID from creation
TASK_ID=$(agentcore task create \
  --type "analysis" \
  --json | jq -r '.result.task_id')

# Wait for completion and get result
agentcore task status $TASK_ID --json | jq '.result.status'
```

### Workflow Definitions

Create workflow YAML files:

```yaml
# workflow.yaml
name: "ci-pipeline"
description: "Continuous integration workflow"
tasks:
  - name: "run-tests"
    type: "testing"
    requirements:
      capability: "testing"

  - name: "build-artifact"
    type: "build"
    requirements:
      capability: "build"
    depends_on:
      - "run-tests"

  - name: "deploy"
    type: "deployment"
    requirements:
      capability: "deployment"
    depends_on:
      - "build-artifact"
```

Execute the workflow:

```bash
agentcore workflow create --file workflow.yaml
agentcore workflow execute <workflow-id> --watch
```

---

## Troubleshooting

### Cannot connect to AgentCore API

```
Error: Cannot connect to AgentCore API at http://localhost:8001
```

**Solutions**:
- Check if AgentCore server is running: `curl http://localhost:8001/health`
- Verify API URL in config: `agentcore config show`
- Update API URL: `agentcore config init` or set `AGENTCORE_API_URL`

### Authentication failed

```
Error: Authentication failed (401 Unauthorized)
```

**Solutions**:
- Check your JWT token: `echo $AGENTCORE_TOKEN`
- Verify token in config: `agentcore config show`
- Set valid token: `export AGENTCORE_TOKEN="your-token"`

### Command not found

```
agentcore: command not found
```

**Solutions**:
- Ensure package is installed: `pip list | grep agentcore`
- Reinstall: `pip install --force-reinstall agentcore`
- Check PATH includes Python scripts directory

### Invalid configuration

```
Error: Invalid configuration file
```

**Solutions**:
- Validate config: `agentcore config validate`
- Check YAML syntax
- Regenerate config: `agentcore config init --force`

---

## Examples

### Example 1: Complete Task Workflow

```bash
# 1. Register an agent
agentcore agent register \
  --name "code-reviewer" \
  --capabilities "python,code-review"

# 2. Create a task
TASK_ID=$(agentcore task create \
  --type "code-review" \
  --description "Review authentication module" \
  --json | jq -r '.result.task_id')

# 3. Watch task execution
agentcore task status $TASK_ID --watch

# 4. Get results
agentcore task result $TASK_ID
```

### Example 2: Multi-Agent Workflow

```bash
# Create workflow definition
cat > ci-workflow.yaml << 'EOF'
name: "ci-pipeline"
tasks:
  - name: "lint"
    type: "linting"
  - name: "test"
    type: "testing"
    depends_on: ["lint"]
  - name: "build"
    type: "build"
    depends_on: ["test"]
EOF

# Execute workflow
agentcore workflow create --file ci-workflow.yaml
agentcore workflow execute <workflow-id> --watch
```

### Example 3: Session Management

```bash
# Start working on a feature
agentcore agent register --name "dev-agent" --capabilities "development"
agentcore task create --type "feature-dev" --description "Add OAuth2"

# Save session
SESSION_ID=$(agentcore session save \
  --name "oauth-feature" \
  --description "OAuth2 implementation" \
  --json | jq -r '.result.session_id')

# Resume later
agentcore session resume $SESSION_ID
```

---

## Performance Tips

1. **Use JSON output for automation**: Faster parsing and machine-readable
2. **Filter results at the source**: Use `--status` and other filters to reduce output
3. **Batch operations**: Group multiple commands in scripts
4. **Cache config**: Set environment variables to avoid repeated config lookups

---

## Development

### Running from Source

```bash
# Clone repository
git clone https://github.com/agentcore/agentcore.git
cd agentcore

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/cli/

# Run linting
ruff check src/agentcore_cli/
mypy src/agentcore_cli/
```

### Building Distribution

```bash
# Build package
python -m build

# Test with TestPyPI
python -m twine upload --repository testpypi dist/*

# Publish to PyPI
python -m twine upload dist/*
```

---

## Contributing

We welcome contributions! See our [Contributing Guide](../../CONTRIBUTING.md) for details.

### Reporting Issues

- **Bug reports**: [GitHub Issues](https://github.com/agentcore/agentcore/issues)
- **Feature requests**: [GitHub Discussions](https://github.com/agentcore/agentcore/discussions)
- **Security issues**: Email security@agentcore.ai

---

## Support

- **Documentation**: [https://docs.agentcore.ai](https://docs.agentcore.ai)
- **Community**: [Discord](https://discord.gg/agentcore)
- **Enterprise Support**: support@agentcore.ai

---

## License

MIT License - see [LICENSE](../../LICENSE) for details.

---

## Changelog

See [CHANGELOG.md](./CHANGELOG.md) for version history and release notes.
