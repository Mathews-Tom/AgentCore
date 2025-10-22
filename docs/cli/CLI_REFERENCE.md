# AgentCore CLI Reference

Complete command reference for the AgentCore CLI.

**Version**: 0.1.0

---

## Table of Contents

- [Global Options](#global-options)
- [Agent Commands](#agent-commands)
- [Task Commands](#task-commands)
- [Session Commands](#session-commands)
- [Workflow Commands](#workflow-commands)
- [Config Commands](#config-commands)

---

## Global Options

Available for all commands:

```bash
agentcore [OPTIONS] COMMAND [ARGS]...
```

| Option | Short | Description |
|--------|-------|-------------|
| `--version` | `-v` | Show version and exit |
| `--help` | | Show help message and exit |

---

## Agent Commands

Manage agent lifecycle and discovery.

### `agentcore agent register`

Register a new agent with AgentCore.

**Usage**:
```bash
agentcore agent register [OPTIONS]
```

**Options**:

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--name` | `-n` | TEXT | Yes | - | Agent name |
| `--capabilities` | `-c` | TEXT | Yes | - | Comma-separated capabilities |
| `--cost-per-request` | | FLOAT | No | 0.01 | Cost per request in USD |
| `--requirements` | `-r` | TEXT | No | - | JSON string of requirements |
| `--json` | | FLAG | No | false | Output in JSON format |

**Examples**:

```bash
# Basic registration
agentcore agent register \
  --name "code-analyzer" \
  --capabilities "python,analysis"

# With requirements
agentcore agent register \
  --name "code-analyzer" \
  --capabilities "python,analysis,linting" \
  --requirements '{"memory": "512MB", "cpu": "0.5"}' \
  --cost-per-request 0.01

# JSON output
agentcore agent register \
  --name "test-agent" \
  --capabilities "testing" \
  --json
```

**Output**:

```
✓ Agent registered successfully
  Agent ID: agent-7f8e9d0c
  Name: code-analyzer
  Status: active
  Capabilities: python, analysis
```

---

### `agentcore agent list`

List all registered agents.

**Usage**:
```bash
agentcore agent list [OPTIONS]
```

**Options**:

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--status` | | TEXT | No | - | Filter by status (active/inactive) |
| `--json` | | FLAG | No | false | Output in JSON format |

**Examples**:

```bash
# List all agents
agentcore agent list

# Filter by status
agentcore agent list --status active

# JSON output
agentcore agent list --json
```

**Output**:

```
                                Agents
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Agent Id       ┃ Name          ┃ Status ┃ Capabilities        ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ agent-7f8e9d0c │ code-analyzer │ active │ python, analysis    │
│ agent-8g9f0e1d │ test-runner   │ active │ testing, python     │
└────────────────┴───────────────┴────────┴─────────────────────┘

Total: 2 agent(s)
```

---

### `agentcore agent info`

Get detailed information about a specific agent.

**Usage**:
```bash
agentcore agent info <agent-id> [OPTIONS]
```

**Arguments**:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `agent-id` | TEXT | Yes | Agent identifier |

**Options**:

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--json` | | FLAG | No | false | Output in JSON format |

**Examples**:

```bash
# Get agent details
agentcore agent info agent-7f8e9d0c

# JSON output
agentcore agent info agent-7f8e9d0c --json
```

**Output**:

```
Agent: code-analyzer (agent-7f8e9d0c)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: active
Capabilities: python, analysis, linting
Cost per Request: $0.01
Requirements:
  - memory: 512MB
  - cpu: 0.5
Registered: 2025-10-21 14:30:00 UTC
Last Active: 2025-10-21 15:45:23 UTC
```

---

### `agentcore agent remove`

Remove an agent from AgentCore.

**Usage**:
```bash
agentcore agent remove <agent-id> [OPTIONS]
```

**Arguments**:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `agent-id` | TEXT | Yes | Agent identifier |

**Options**:

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--force` | `-f` | FLAG | No | false | Skip confirmation prompt |

**Examples**:

```bash
# Remove agent (with confirmation)
agentcore agent remove agent-7f8e9d0c

# Force remove (no confirmation)
agentcore agent remove agent-7f8e9d0c --force
```

**Output**:

```
⚠️  Remove agent 'code-analyzer' (agent-7f8e9d0c)?
   This action cannot be undone.

   Continue? [y/N]: y

✓ Agent removed successfully
```

---

### `agentcore agent search`

Search for agents by capability.

**Usage**:
```bash
agentcore agent search [OPTIONS]
```

**Options**:

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--capability` | `-c` | TEXT | Yes | - | Capability to search for (can be used multiple times) |
| `--json` | | FLAG | No | false | Output in JSON format |

**Examples**:

```bash
# Search for single capability
agentcore agent search --capability "python"

# Search for multiple capabilities (AND)
agentcore agent search \
  --capability "python" \
  --capability "testing"

# JSON output
agentcore agent search --capability "python" --json
```

**Output**:

```
                          Matching Agents
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Agent Id       ┃ Name          ┃ Capabilities             ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ agent-7f8e9d0c │ code-analyzer │ python, analysis         │
│ agent-8g9f0e1d │ test-runner   │ python, testing, pytest  │
└────────────────┴───────────────┴──────────────────────────┘

Total: 2 matching agent(s)
```

---

## Task Commands

Create, monitor, and manage tasks.

### `agentcore task create`

Create a new task.

**Usage**:
```bash
agentcore task create [OPTIONS]
```

**Options**:

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--type` | `-t` | TEXT | Yes | - | Task type |
| `--description` | `-d` | TEXT | No | - | Task description |
| `--requirements` | `-r` | TEXT | No | - | JSON string of requirements |
| `--priority` | `-p` | TEXT | No | medium | Priority (low/medium/high/critical) |
| `--json` | | FLAG | No | false | Output in JSON format |

**Examples**:

```bash
# Basic task
agentcore task create \
  --type "code-review" \
  --description "Review authentication module"

# With requirements and priority
agentcore task create \
  --type "code-review" \
  --description "Review critical security code" \
  --requirements '{"language": "python"}' \
  --priority high

# JSON output (for scripting)
TASK_ID=$(agentcore task create \
  --type "analysis" \
  --json | jq -r '.result.task_id')
```

**Output**:

```
✓ Task created successfully
  Task ID: task-a1b2c3d4
  Type: code-review
  Status: pending
  Priority: medium
  Description: Review authentication module
```

---

### `agentcore task status`

Get task status.

**Usage**:
```bash
agentcore task status <task-id> [OPTIONS]
```

**Arguments**:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `task-id` | TEXT | Yes | Task identifier |

**Options**:

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--watch` | `-w` | FLAG | No | false | Watch status updates in real-time |
| `--interval` | | INT | No | 5 | Watch interval in seconds |
| `--json` | | FLAG | No | false | Output in JSON format |

**Examples**:

```bash
# Get current status
agentcore task status task-a1b2c3d4

# Watch in real-time
agentcore task status task-a1b2c3d4 --watch

# Custom watch interval
agentcore task status task-a1b2c3d4 --watch --interval 2

# JSON output
agentcore task status task-a1b2c3d4 --json
```

**Output**:

```
Task: task-a1b2c3d4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Type: code-review
Status: running
Priority: medium
Agent: agent-7f8e9d0c (code-analyzer)
Progress: 45%
Started: 2025-10-21 15:30:00 UTC
Elapsed: 2m 30s
```

---

### `agentcore task list`

List tasks.

**Usage**:
```bash
agentcore task list [OPTIONS]
```

**Options**:

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--status` | | TEXT | No | - | Filter by status |
| `--limit` | | INT | No | 10 | Maximum number of tasks to show |
| `--json` | | FLAG | No | false | Output in JSON format |

**Examples**:

```bash
# List recent tasks
agentcore task list

# Filter by status
agentcore task list --status running

# Limit results
agentcore task list --limit 5

# JSON output
agentcore task list --json
```

**Output**:

```
                                     Tasks
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Task Id        ┃ Type        ┃ Status  ┃ Priority┃ Created        ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ task-a1b2c3d4  │ code-review │ running │ medium  │ 2m ago         │
│ task-b2c3d4e5  │ testing     │ pending │ high    │ 5m ago         │
│ task-c3d4e5f6  │ analysis    │ completed│ low    │ 15m ago        │
└────────────────┴─────────────┴─────────┴─────────┴────────────────┘

Total: 3 task(s)
```

---

### `agentcore task cancel`

Cancel a running task.

**Usage**:
```bash
agentcore task cancel <task-id>
```

**Arguments**:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `task-id` | TEXT | Yes | Task identifier |

**Examples**:

```bash
agentcore task cancel task-a1b2c3d4
```

**Output**:

```
✓ Task cancelled successfully
  Task ID: task-a1b2c3d4
  Status: cancelled
```

---

### `agentcore task result`

Get task result and artifacts.

**Usage**:
```bash
agentcore task result <task-id> [OPTIONS]
```

**Arguments**:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `task-id` | TEXT | Yes | Task identifier |

**Options**:

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--output` | `-o` | PATH | No | - | Save result to file |
| `--json` | | FLAG | No | false | Output in JSON format |

**Examples**:

```bash
# Display result
agentcore task result task-a1b2c3d4

# Save to file
agentcore task result task-a1b2c3d4 --output result.json

# JSON output
agentcore task result task-a1b2c3d4 --json
```

**Output**:

```
Task Result: task-a1b2c3d4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: completed
Completed: 2025-10-21 15:45:30 UTC
Duration: 5m 30s

Result:
{
  "issues_found": 3,
  "severity": "medium",
  "recommendations": [
    "Add type hints to function parameters",
    "Use context managers for file operations",
    "Add docstrings to public methods"
  ]
}

Artifacts:
  - report.html (15.2 KB)
  - analysis.json (2.4 KB)
```

---

### `agentcore task retry`

Retry a failed task.

**Usage**:
```bash
agentcore task retry <task-id>
```

**Arguments**:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `task-id` | TEXT | Yes | Task identifier |

**Examples**:

```bash
agentcore task retry task-a1b2c3d4
```

**Output**:

```
✓ Task retry initiated
  Task ID: task-a1b2c3d4
  Status: pending
  Previous attempts: 1
```

---

## Session Commands

Save and resume long-running workflows.

### `agentcore session save`

Save current workflow session.

**Usage**:
```bash
agentcore session save [OPTIONS]
```

**Options**:

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--name` | `-n` | TEXT | Yes | - | Session name |
| `--description` | `-d` | TEXT | No | - | Session description |
| `--tags` | `-t` | TEXT | No | - | Comma-separated tags |
| `--json` | | FLAG | No | false | Output in JSON format |

**Examples**:

```bash
# Basic save
agentcore session save --name "feature-dev"

# With description and tags
agentcore session save \
  --name "oauth-feature" \
  --description "OAuth2 implementation" \
  --tags "auth,backend,security"

# JSON output
SESSION_ID=$(agentcore session save \
  --name "my-session" \
  --json | jq -r '.result.session_id')
```

**Output**:

```
✓ Session saved successfully
  Session ID: session-x1y2z3
  Name: oauth-feature
  Tasks saved: 5
  Agents saved: 2

  Resume with: agentcore session resume session-x1y2z3
```

---

### `agentcore session resume`

Resume a saved session.

**Usage**:
```bash
agentcore session resume <session-id>
```

**Arguments**:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `session-id` | TEXT | Yes | Session identifier |

**Examples**:

```bash
agentcore session resume session-x1y2z3
```

**Output**:

```
Restoring session... ━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%

✓ Session resumed successfully
  Session ID: session-x1y2z3
  Name: oauth-feature
  Tasks restored: 5
  Agents restored: 2
```

---

### `agentcore session list`

List sessions.

**Usage**:
```bash
agentcore session list [OPTIONS]
```

**Options**:

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--status` | | TEXT | No | - | Filter by status |
| `--json` | | FLAG | No | false | Output in JSON format |

**Examples**:

```bash
# List all sessions
agentcore session list

# Filter by status
agentcore session list --status active

# JSON output
agentcore session list --json
```

**Output**:

```
                                    Sessions
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Session Id     ┃ Name          ┃ Status ┃ Tasks   ┃ Created        ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ session-x1y2z3 │ oauth-feature │ active │ 5       │ 1h ago         │
│ session-y2z3a4 │ testing-suite │ paused │ 3       │ 2h ago         │
└────────────────┴───────────────┴────────┴─────────┴────────────────┘

Total: 2 session(s)
```

---

### `agentcore session info`

Get session details.

**Usage**:
```bash
agentcore session info <session-id> [OPTIONS]
```

**Arguments**:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `session-id` | TEXT | Yes | Session identifier |

**Options**:

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--json` | | FLAG | No | false | Output in JSON format |

**Examples**:

```bash
agentcore session info session-x1y2z3
```

**Output**:

```
Session: oauth-feature (session-x1y2z3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: active
Description: OAuth2 implementation
Tags: auth, backend, security

Tasks: 5
  - 3 completed
  - 1 running
  - 1 pending

Agents: 2
  - agent-7f8e9d0c (code-analyzer)
  - agent-8g9f0e1d (test-runner)

Created: 2025-10-21 14:00:00 UTC
Last Updated: 2025-10-21 15:30:00 UTC
```

---

### `agentcore session delete`

Delete a session.

**Usage**:
```bash
agentcore session delete <session-id>
```

**Arguments**:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `session-id` | TEXT | Yes | Session identifier |

**Examples**:

```bash
agentcore session delete session-x1y2z3
```

**Output**:

```
✓ Session deleted successfully
```

---

### `agentcore session export`

Export session for debugging.

**Usage**:
```bash
agentcore session export <session-id> [OPTIONS]
```

**Arguments**:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `session-id` | TEXT | Yes | Session identifier |

**Options**:

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--output` | `-o` | PATH | No | - | Output file path |

**Examples**:

```bash
# Export to file
agentcore session export session-x1y2z3 --output session.json

# Output to stdout
agentcore session export session-x1y2z3
```

---

## Workflow Commands

Define and execute multi-step workflows.

### `agentcore workflow create`

Create workflow from YAML definition file.

**Usage**:
```bash
agentcore workflow create [OPTIONS]
```

**Options**:

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--file` | `-f` | PATH | Yes | - | Workflow YAML file path |
| `--json` | | FLAG | No | false | Output in JSON format |

**Examples**:

```bash
# Create workflow
agentcore workflow create --file workflow.yaml

# JSON output
WORKFLOW_ID=$(agentcore workflow create \
  --file workflow.yaml \
  --json | jq -r '.result.workflow_id')
```

**Workflow YAML Format**:

```yaml
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
```

**Output**:

```
✓ Workflow created successfully
  Workflow ID: workflow-p1q2r3
  Name: ci-pipeline
  Tasks: 2
```

---

### `agentcore workflow execute`

Execute a workflow.

**Usage**:
```bash
agentcore workflow execute <workflow-id> [OPTIONS]
```

**Arguments**:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `workflow-id` | TEXT | Yes | Workflow identifier |

**Options**:

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--watch` | `-w` | FLAG | No | false | Watch execution in real-time |
| `--json` | | FLAG | No | false | Output in JSON format |

**Examples**:

```bash
# Execute workflow
agentcore workflow execute workflow-p1q2r3

# Watch execution
agentcore workflow execute workflow-p1q2r3 --watch
```

**Output**:

```
Workflow: ci-pipeline (workflow-p1q2r3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ run-tests      [completed] 2m 30s
⏵ build-artifact [running]   45s
⏸ deploy         [pending]
```

---

### `agentcore workflow status`

Get workflow status showing task progress.

**Usage**:
```bash
agentcore workflow status <workflow-id> [OPTIONS]
```

**Arguments**:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `workflow-id` | TEXT | Yes | Workflow identifier |

**Options**:

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--json` | | FLAG | No | false | Output in JSON format |

**Examples**:

```bash
agentcore workflow status workflow-p1q2r3
```

---

### `agentcore workflow list`

List workflows.

**Usage**:
```bash
agentcore workflow list [OPTIONS]
```

**Options**:

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--status` | | TEXT | No | - | Filter by status |
| `--json` | | FLAG | No | false | Output in JSON format |

**Examples**:

```bash
# List all workflows
agentcore workflow list

# Filter by status
agentcore workflow list --status running
```

---

### `agentcore workflow visualize`

Visualize workflow as ASCII graph.

**Usage**:
```bash
agentcore workflow visualize <workflow-id>
```

**Arguments**:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `workflow-id` | TEXT | Yes | Workflow identifier |

**Examples**:

```bash
agentcore workflow visualize workflow-p1q2r3
```

**Output**:

```
Workflow: ci-pipeline
├── Task: run-tests
│   ├── Status: completed
│   ├── Agent: test-runner
│   └── Duration: 2m 30s
├── Task: build-artifact
│   ├── Status: running
│   ├── Agent: build-agent
│   └── Progress: 45%
└── Task: deploy
    └── Status: pending
```

---

### `agentcore workflow pause`

Pause a running workflow.

**Usage**:
```bash
agentcore workflow pause <workflow-id>
```

**Arguments**:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `workflow-id` | TEXT | Yes | Workflow identifier |

**Examples**:

```bash
agentcore workflow pause workflow-p1q2r3
```

---

### `agentcore workflow resume`

Resume a paused workflow.

**Usage**:
```bash
agentcore workflow resume <workflow-id>
```

**Arguments**:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `workflow-id` | TEXT | Yes | Workflow identifier |

**Examples**:

```bash
agentcore workflow resume workflow-p1q2r3
```

---

## Config Commands

Manage CLI configuration.

### `agentcore config init`

Initialize configuration file with default template.

**Usage**:
```bash
agentcore config init [OPTIONS]
```

**Options**:

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--global` | `-g` | FLAG | No | false | Create global config (~/.agentcore/config.yaml) |
| `--force` | `-f` | FLAG | No | false | Overwrite existing config |

**Examples**:

```bash
# Create project config
agentcore config init

# Create global config
agentcore config init --global

# Overwrite existing
agentcore config init --force
```

**Output**:

```
✓ Configuration file created: ./.agentcore.yaml

Edit the file to customize settings, then run:
  agentcore config validate
```

---

### `agentcore config show`

Display current configuration with merged values.

**Usage**:
```bash
agentcore config show [OPTIONS]
```

**Options**:

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--sources` | | FLAG | No | false | Show config sources and precedence |
| `--json` | | FLAG | No | false | Output in JSON format |

**Examples**:

```bash
# Show current config
agentcore config show

# Show with sources
agentcore config show --sources

# JSON output
agentcore config show --json
```

**Output**:

```
Current Configuration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

API:
  URL: http://localhost:8001
  Timeout: 30s
  Retries: 3
  Verify SSL: true

Authentication:
  Type: jwt
  Token: eyJhbGc... (set via environment)

Output:
  Format: table
  Color: true
  Timestamps: false
  Verbose: false
```

---

### `agentcore config validate`

Validate configuration file syntax and values.

**Usage**:
```bash
agentcore config validate
```

**Examples**:

```bash
agentcore config validate
```

**Output**:

```
✓ Configuration is valid

Config file: ./.agentcore.yaml
Schema version: 1.0
All settings validated successfully
```

---

## Exit Codes

The CLI uses the following exit codes:

| Code | Meaning | Description |
|------|---------|-------------|
| 0 | Success | Command completed successfully |
| 1 | General error | API error, command failed |
| 2 | Usage error | Invalid arguments, missing required fields |
| 3 | Connection error | Cannot connect to API server |
| 4 | Authentication error | Invalid or expired credentials |
| 5 | Timeout error | Operation exceeded timeout |
| 130 | Interrupted | User cancelled operation (Ctrl+C) |

---

## Environment Variables

All configuration options can be set via environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `AGENTCORE_API_URL` | API server URL | `http://localhost:8001` |
| `AGENTCORE_API_TIMEOUT` | Request timeout (seconds) | `30` |
| `AGENTCORE_TOKEN` | JWT authentication token | `eyJhbGc...` |
| `AGENTCORE_API_KEY` | API key | `ak_...` |
| `AGENTCORE_OUTPUT_FORMAT` | Default output format | `json` |
| `AGENTCORE_COLOR` | Enable color output | `true` |
| `AGENTCORE_VERBOSE` | Verbose output | `false` |
| `AGENTCORE_VERIFY_SSL` | Verify SSL certificates | `true` |

---

## Notes

- All timestamps are in UTC
- JSON output is always pretty-printed
- Table widths auto-adjust to terminal size
- Color output is disabled when piping to files or other commands
- Watch mode can be interrupted with `Ctrl+C` without stopping the task/workflow

---

**Last Updated**: 2025-10-21
**Version**: 0.1.0
