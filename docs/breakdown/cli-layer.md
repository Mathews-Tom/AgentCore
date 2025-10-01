# Technical Breakdown: CLI Layer

**Component:** CLI Layer (Command Line Interface)
**Version:** 1.0
**Status:** Design Phase
**Last Updated:** 2025-10-01
**Owner:** TBD (Mid-level Developer)

---

## Quick Reference

```yaml
complexity: Medium
risk_level: Low-Medium
team_size: 1 mid-level developer
duration: 4 weeks (2 sprints)
story_points: 34 SP

dependencies:
  runtime:
    - Python 3.12+
    - Typer 0.9+ (CLI framework)
    - Rich 13+ (output formatting)
    - PyYAML 6+ (configuration)
    - httpx 0.24+ (async HTTP client)
    - Pydantic 2.0+ (validation)
  external:
    - AgentCore A2A Protocol API (JSON-RPC 2.0)
    - Session Management API (A2A-019, A2A-020) - Phase 5 dependency

key_risks:
  - Session API not yet implemented (blocks CLI-004, CLI-005)
  - JSON-RPC client complexity
  - Configuration precedence bugs

performance_targets:
  - Command response time: <2s for simple commands
  - API call overhead: <500ms
  - Output rendering: <100ms for 1000 rows

testing:
  - Unit test coverage: 85%+
  - Integration tests with mock API
  - E2E tests with testcontainers

phases:
  - Phase 1: Core CLI Framework (Week 1)
  - Phase 2: Agent & Task Commands (Week 2)
  - Phase 3: Session & Workflow Commands (Week 3)
  - Phase 4: Polish & Documentation (Week 4)
```

---

## 1. Component Overview

### 1.1 Purpose

The CLI Layer provides a command-line interface for interacting with the AgentCore platform. It offers developers and operators an efficient way to:

- Register and manage AI agents
- Create and monitor tasks
- Save and resume workflow sessions
- Execute complex workflows
- Configure AgentCore settings

**Strategic Importance:** CLI is the primary developer interface for AgentCore, critical for onboarding, debugging, and operational workflows. High-quality CLI experience directly impacts developer satisfaction and platform adoption.

### 1.2 Scope

**In Scope:**

- Command-line interface with 5 command groups (agent, task, session, workflow, config)
- JSON-RPC 2.0 client for AgentCore API communication
- Multi-level configuration system (CLI args > env vars > project/global config > defaults)
- Output formatting (table, JSON, tree) with Rich library
- Authentication token management
- Error handling and user-friendly messages
- Basic shell completions (bash, zsh, fish)

**Out of Scope:**

- Web UI or GUI (separate component)
- Agent implementation (runtime concern)
- Direct database access (all via API)
- Real-time event streaming (basic polling only)
- Interactive TUI (future enhancement)

**Success Criteria:**

- All 12 CLI tasks completed (34 SP)
- 85%+ test coverage
- Command response time <2s for simple operations
- Clear, actionable error messages for all failure scenarios
- Shell completion support for all commands

### 1.3 Architecture Context

```text
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
│                   User / Developer                      │
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                             │
                             │ Shell Commands
                             ↓
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
│                      CLI Layer                          │
┏━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━┓
│  Command      │  JSON-RPC    │  Output                  │
│  Parser       │  Client      │  Formatter               │
│  (Typer)      │  (httpx)     │  (Rich)                  │
┗━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━┓
│       ↓       │       ↓      │       ↓                  │
┗━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━┛
│  Config       │  Auth        │  Error                   │
│  Manager      │  Manager     │  Handler                 │
┗━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                             │
                             │ HTTPS + JSON-RPC 2.0
                             ↓
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
│           AgentCore A2A Protocol Layer                  │
│           (FastAPI + JSON-RPC Handler)                  │
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                             │
                             ↓
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
│        PostgreSQL + Redis + Vector Database             │
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

**Key Architectural Decisions:**

1. **Stateless Design:** CLI maintains no local state; all data managed by AgentCore API
2. **Configuration Hierarchy:** 5-level precedence (CLI > env > project > global > defaults)
3. **JSON-RPC Client:** Custom async client with retry logic, not REST (aligns with A2A protocol)
4. **Output Flexibility:** Support table, JSON, tree formats for different use cases
5. **Authentication:** JWT token stored in system keychain (secure) with plaintext fallback

---

## 2. System Context

### 2.1 External Dependencies

```yaml
AgentCore A2A Protocol API:
  protocol: JSON-RPC 2.0 over HTTPS
  endpoint: http://localhost:8001/api/v1/jsonrpc (default)
  authentication: JWT Bearer token
  methods:
    - agent.register
    - agent.list
    - agent.get
    - agent.delete
    - task.create
    - task.get
    - task.list
    - task.update
    - session.save (Phase 5 - A2A-019)
    - session.resume (Phase 5 - A2A-020)
    - session.list (Phase 5 - A2A-021)
    - session.info (Phase 5 - A2A-021)
    - workflow.create
    - workflow.execute
  availability: 99.9% SLA

System Keychain:
  library: keyring (Python)
  purpose: Secure JWT token storage
  platforms: macOS Keychain, Windows Credential Manager, Linux Secret Service
  fallback: ~/.agentcore/token (plaintext with warning)
```

### 2.2 Integration Points

**Configuration Files:**

- Global: `~/.agentcore/config.yaml`
- Project: `./.agentcore.yaml`
- Format: YAML with Pydantic validation
- Environment: `AGENTCORE_*` prefixed variables

**Logging:**

- Location: `~/.agentcore/logs/cli.log`
- Format: Structured JSON logs
- Rotation: Keep last 7 days
- Levels: DEBUG, INFO, WARNING, ERROR

**Shell Completions:**

- Auto-generated by Typer
- Supported: bash, zsh, fish, powershell
- Installation: `agentcore --install-completion`

---

## 3. Architecture Design

### 3.1 Module Structure

```text
src/agentcore_cli/
├── __init__.py                 # Package initialization
├── main.py                     # Entry point, Typer app setup
├── commands/
│   ├── __init__.py
│   ├── agent.py                # Agent commands (register, list, get, delete)
│   ├── task.py                 # Task commands (create, get, list, update)
│   ├── session.py              # Session commands (save, resume, list, info)
│   ├── workflow.py             # Workflow commands (create, execute, status)
│   └── config.py               # Config commands (init, show, validate)
├── client/
│   ├── __init__.py
│   ├── jsonrpc_client.py       # JSON-RPC 2.0 client implementation
│   ├── auth.py                 # Authentication token management
│   └── retry.py                # Retry logic with exponential backoff
├── config/
│   ├── __init__.py
│   ├── models.py               # Pydantic config models
│   ├── loader.py               # Multi-level config loading
│   └── defaults.py             # Default configuration values
├── output/
│   ├── __init__.py
│   ├── formatters.py           # Table, JSON, tree formatters
│   ├── console.py              # Rich console wrapper
│   └── themes.py               # Color themes and styles
├── utils/
│   ├── __init__.py
│   ├── errors.py               # Custom exceptions and error handling
│   ├── validation.py           # Input validation helpers
│   └── logging.py              # Logging configuration
└── version.py                  # Version information

tests/
├── unit/
│   ├── test_jsonrpc_client.py
│   ├── test_config_loader.py
│   ├── test_formatters.py
│   └── test_auth.py
├── integration/
│   ├── test_agent_commands.py
│   ├── test_task_commands.py
│   ├── test_session_commands.py
│   └── test_workflow_commands.py
└── e2e/
    └── test_full_workflows.py
```

**Complexity Assessment:**

| Module | Lines | Complexity | Risk |
|--------|-------|------------|------|
| `jsonrpc_client.py` | 300 | High | Medium - Retry logic, error mapping |
| `config/loader.py` | 200 | Medium | High - Config precedence bugs likely |
| `commands/agent.py` | 250 | Low | Low - Simple CRUD operations |
| `commands/task.py` | 200 | Low | Low - Simple CRUD operations |
| `commands/session.py` | 150 | Medium | High - Depends on unbuilt API |
| `commands/workflow.py` | 180 | Medium | Medium - Complex workflows |
| `output/formatters.py` | 150 | Low | Low - Display logic only |
| `auth.py` | 100 | Low | Low - Keyring wrapper |
| **Total** | **~1800** | **Medium** | **Medium** |

### 3.2 Data Flow

**Typical Command Execution Flow:**

```
1. User types command:
   $ agentcore agent register --name "MyAgent" --url "http://localhost:9000"

2. Typer parses command and arguments
   ↓
3. Config loader merges configurations:
   CLI args > env vars > project config > global config > defaults
   ↓
4. Auth manager loads JWT token from keychain
   ↓
5. JSON-RPC client constructs request:
   {
     "jsonrpc": "2.0",
     "method": "agent.register",
     "params": {
       "name": "MyAgent",
       "base_url": "http://localhost:9000"
     },
     "id": "req-001"
   }
   ↓
6. HTTP client sends POST to API with retries (3 attempts, exponential backoff)
   ↓
7. API returns JSON-RPC response:
   {
     "jsonrpc": "2.0",
     "result": {
       "agent_id": "agt_123",
       "status": "active",
       "created_at": "2025-10-01T12:00:00Z"
     },
     "id": "req-001"
   }
   ↓
8. Client validates response, extracts result
   ↓
9. Output formatter renders result:
   [Table format]
   ┏━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
   ┃Agent ID┃Name     ┃Status  ┃Created At           ┃
   ┣━━━━━━━━╋━━━━━━━━━╋━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━┫
   ┃agt_123 ┃MyAgent  ┃active  ┃2025-10-01 12:00:00  ┃
   ┗━━━━━━━━┻━━━━━━━━━┻━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━┛
   ↓
10. Exit code 0 (success)
```

**Error Handling Flow:**

```
1. API error (e.g., agent already exists)
   ↓
2. JSON-RPC error response:
   {
     "jsonrpc": "2.0",
     "error": {
       "code": -32000,
       "message": "Agent already registered",
       "data": {"agent_id": "agt_123"}
     },
     "id": "req-001"
   }
   ↓
3. Client maps JSON-RPC error to CLI exception
   ↓
4. Error handler displays user-friendly message:
   [Error] Agent already registered

   An agent with this name already exists:
   - Agent ID: agt_123
   - Name: MyAgent

   Use 'agentcore agent get agt_123' to view details.
   ↓
5. Exit code 1 (error)
```

---

## 4. Interface Contracts

### 4.1 Command Groups

#### 4.1.1 Agent Commands

**Command:** `agentcore agent register`

```bash
# Usage
agentcore agent register [OPTIONS]

# Options
--name TEXT              Agent name [required]
--url TEXT               Agent base URL [required]
--description TEXT       Agent description [optional]
--capabilities TEXT      Comma-separated capabilities [optional]
--output FORMAT          Output format: table|json|tree [default: table]
--help                   Show help message

# Example
$ agentcore agent register \
    --name "CodeAnalyzer" \
    --url "http://localhost:9000" \
    --description "Static code analysis agent" \
    --capabilities "code_review,linting,security_scan"

# Output (table format)
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃Agent ID    ┃Name         ┃Status ┃Created At           ┃
┣━━━━━━━━━━━━╋━━━━━━━━━━━━━╋━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━┫
┃agt_abc123  ┃CodeAnalyzer ┃active ┃2025-10-01 14:30:00  ┃
┗━━━━━━━━━━━━┻━━━━━━━━━━━━━┻━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━┛

# JSON-RPC Method
agent.register

# Request
{
  "jsonrpc": "2.0",
  "method": "agent.register",
  "params": {
    "name": "CodeAnalyzer",
    "base_url": "http://localhost:9000",
    "description": "Static code analysis agent",
    "capabilities": ["code_review", "linting", "security_scan"]
  },
  "id": "req-001"
}

# Response
{
  "jsonrpc": "2.0",
  "result": {
    "agent_id": "agt_abc123",
    "name": "CodeAnalyzer",
    "status": "active",
    "created_at": "2025-10-01T14:30:00Z"
  },
  "id": "req-001"
}
```

**Command:** `agentcore agent list`

```bash
# Usage
agentcore agent list [OPTIONS]

# Options
--status TEXT            Filter by status: active|inactive|all [default: all]
--capability TEXT        Filter by capability
--limit INT              Max results [default: 50]
--output FORMAT          Output format [default: table]
--help                   Show help

# Example
$ agentcore agent list --status active --limit 10

# Output
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┓
┃Agent ID    ┃Name           ┃Status ┃Capabilities ┃
┣━━━━━━━━━━━━╋━━━━━━━━━━━━━━━╋━━━━━━━╋━━━━━━━━━━━━━┫
┃agt_abc123  ┃CodeAnalyzer   ┃active ┃code_review  ┃
┃agt_def456  ┃TestRunner     ┃active ┃testing      ┃
┗━━━━━━━━━━━━┻━━━━━━━━━━━━━━━┻━━━━━━━┻━━━━━━━━━━━━━┛

Showing 2 of 2 agents
```

**Command:** `agentcore agent get <agent_id>`

```bash
# Usage
agentcore agent get <agent_id> [OPTIONS]

# Options
--output FORMAT          Output format [default: tree]
--help                   Show help

# Example
$ agentcore agent get agt_abc123

# Output (tree format)
Agent: agt_abc123
├── Name: CodeAnalyzer
├── Status: active
├── Base URL: http://localhost:9000
├── Description: Static code analysis agent
├── Capabilities
│   ├── code_review
│   ├── linting
│   └── security_scan
├── Health
│   ├── Status: healthy
│   ├── Last Check: 2025-10-01 14:35:00
│   └── Response Time: 45ms
└── Created: 2025-10-01 14:30:00
```

**Command:** `agentcore agent delete <agent_id>`

```bash
# Usage
agentcore agent delete <agent_id> [OPTIONS]

# Options
--force                  Skip confirmation prompt
--help                   Show help

# Example
$ agentcore agent delete agt_abc123

# Output
Are you sure you want to delete agent 'CodeAnalyzer' (agt_abc123)? [y/N]: y
✓ Agent deleted successfully
```

#### 4.1.2 Task Commands

**Command:** `agentcore task create`

```bash
# Usage
agentcore task create [OPTIONS]

# Options
--description TEXT       Task description [required]
--agent TEXT             Target agent ID or capability [required]
--priority INT           Priority 1-5 [default: 3]
--timeout INT            Timeout in seconds [optional]
--input-file PATH        Input data file (JSON) [optional]
--output FORMAT          Output format [default: table]
--help                   Show help

# Example
$ agentcore task create \
    --description "Review pull request #123" \
    --agent "code_review" \
    --priority 2 \
    --input-file pr_data.json

# Output
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┓
┃Task ID    ┃Description              ┃Agent      ┃Status  ┃
┣━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━┫
┃tsk_xyz789 ┃Review pull request      ┃agt_abc123 ┃pending ┃
┗━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━┻━━━━━━━━┛

Task created successfully
Use 'agentcore task get tsk_xyz789' to check status
```

**Command:** `agentcore task get <task_id>`

```bash
# Usage
agentcore task get <task_id> [OPTIONS]

# Options
--output FORMAT          Output format [default: tree]
--show-artifacts         Show task artifacts
--help                   Show help

# Example
$ agentcore task get tsk_xyz789 --show-artifacts

# Output (tree format)
Task: tsk_xyz789
├── Description: Review pull request #123
├── Status: completed
├── Agent: agt_abc123 (CodeAnalyzer)
├── Priority: 2 (high)
├── Progress: 100%
├── Created: 2025-10-01 15:00:00
├── Started: 2025-10-01 15:00:15
├── Completed: 2025-10-01 15:05:32
├── Duration: 5m 17s
└── Artifacts
    ├── review_report.md (12.5 KB)
    └── issues.json (3.2 KB)

Use 'agentcore task artifact tsk_xyz789 review_report.md' to download
```

**Command:** `agentcore task list`

```bash
# Usage
agentcore task list [OPTIONS]

# Options
--status TEXT            Filter by status: pending|running|completed|failed [default: all]
--agent TEXT             Filter by agent ID
--limit INT              Max results [default: 50]
--output FORMAT          Output format [default: table]
--help                   Show help

# Example
$ agentcore task list --status running

# Output
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃Task ID    ┃Description          ┃Agent      ┃Status  ┃Progress  ┃
┣━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━╋━━━━━━━━━━┫
┃tsk_aaa111 ┃Analyze codebase     ┃agt_abc123 ┃running ┃45%       ┃
┃tsk_bbb222 ┃Run security scan    ┃agt_def456 ┃running ┃78%       ┃
┗━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━┻━━━━━━━━┻━━━━━━━━━━┛

Showing 2 running tasks
```

#### 4.1.3 Session Commands

**Command:** `agentcore session save`

```bash
# Usage
agentcore session save [OPTIONS]

# Options
--name TEXT              Session name [required]
--description TEXT       Session description [optional]
--tags TEXT              Comma-separated tags [optional]
--output FORMAT          Output format [default: table]
--help                   Show help

# Example
$ agentcore session save \
    --name "pr-review-checkpoint" \
    --description "Checkpoint after code analysis" \
    --tags "code-review,checkpoint"

# Output
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃Session ID           ┃Name                ┃Tasks      ┃Created At  ┃
┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━━━┫
┃ses_123abc           ┃pr-review-checkpoint┃5 captured ┃15:30:00    ┃
┗━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━┻━━━━━━━━━━━━┛

✓ Session saved successfully (snapshot size: 2.3 MB)
Use 'agentcore session resume ses_123abc' to restore

# JSON-RPC Method
session.save

# Dependency
Requires A2A-019 (Session Snapshot Creation) - Phase 5
```

**Command:** `agentcore session resume <session_id>`

```bash
# Usage
agentcore session resume <session_id> [OPTIONS]

# Options
--replay-events          Replay event history [default: false]
--output FORMAT          Output format [default: table]
--help                   Show help

# Example
$ agentcore session resume ses_123abc

# Output
Resuming session 'pr-review-checkpoint'...
   Loading snapshot (2.3 MB)... ✓
   Restoring 5 tasks... ✓
   Reconnecting 2 agents... ✓
   Validating state... ✓

✓ Session resumed successfully in 847ms

Task Status:
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃Task ID     ┃Description          ┃Status  ┃
┣━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━┫
┃tsk_aaa111  ┃Analyze codebase     ┃resumed ┃
┃tsk_bbb222  ┃Review architecture  ┃resumed ┃
┗━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━┛

# JSON-RPC Method
session.resume

# Dependency
Requires A2A-020 (Session Resumption) - Phase 5
```

**Command:** `agentcore session list`

```bash
# Usage
agentcore session list [OPTIONS]

# Options
--state TEXT             Filter by state: PAUSED|RESUMED|COMPLETED [default: all]
--tags TEXT              Filter by tags
--limit INT              Max results [default: 50]
--output FORMAT          Output format [default: table]
--help                   Show help

# Example
$ agentcore session list --state PAUSED

# Output
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃Session ID     ┃Name                ┃Tasks  ┃Created     ┃
┣━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━╋━━━━━━━╋━━━━━━━━━━━━┫
┃ses_123abc     ┃pr-review-checkpoint┃5      ┃1 hour ago  ┃
┃ses_456def     ┃refactor-analysis   ┃8      ┃2 days ago  ┃
┗━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━┻━━━━━━━┻━━━━━━━━━━━━┛

# JSON-RPC Method
session.list

# Dependency
Requires A2A-021 (Session Management API) - Phase 5
```

#### 4.1.4 Workflow Commands

**Command:** `agentcore workflow create`

```bash
# Usage
agentcore workflow create [OPTIONS]

# Options
--file PATH              Workflow definition file (YAML) [required]
--name TEXT              Override workflow name [optional]
--validate-only          Only validate, don't create [default: false]
--output FORMAT          Output format [default: table]
--help                   Show help

# Example
$ agentcore workflow create --file ci_pipeline.yaml

# Output
Validating workflow... ✓
Creating workflow 'CI Pipeline'... ✓

┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃Workflow ID   ┃Name         ┃Tasks  ┃Pattern     ┃
┣━━━━━━━━━━━━━╋━━━━━━━━━━━━━╋━━━━━━━╋━━━━━━━━━━━━┫
┃wf_789ghi     ┃CI Pipeline  ┃6      ┃supervisor  ┃
┗━━━━━━━━━━━━━┻━━━━━━━━━━━━━┻━━━━━━━┻━━━━━━━━━━━━┛

Use 'agentcore workflow execute wf_789ghi' to run
```

**Command:** `agentcore workflow execute <workflow_id>`

```bash
# Usage
agentcore workflow execute <workflow_id> [OPTIONS]

# Options
--input-file PATH        Input data file (JSON) [optional]
--watch                  Watch execution in real-time [default: false]
--output FORMAT          Output format [default: table]
--help                   Show help

# Example
$ agentcore workflow execute wf_789ghi --watch

# Output
Executing workflow 'CI Pipeline'...

Tasks:
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃Task ID     ┃Description        ┃Status  ┃Progress  ┃
┣━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━╋━━━━━━━━╋━━━━━━━━━━┫
┃tsk_001     ┃Checkout code      ┃done    ┃100%      ┃
┃tsk_002     ┃Run tests          ┃running ┃67%       ┃
┃tsk_003     ┃Build artifacts    ┃pending ┃0%        ┃
┃tsk_004     ┃Deploy staging     ┃pending ┃0%        ┃
┗━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━┻━━━━━━━━┻━━━━━━━━━━┛

[Live updates every 2s... Press Ctrl+C to stop watching]
```

#### 4.1.5 Config Commands

**Command:** `agentcore config init`

```bash
# Usage
agentcore config init [OPTIONS]

# Options
--global                 Create global config [default: project config]
--force                  Overwrite existing config
--interactive            Interactive configuration wizard
--help                   Show help

# Example
$ agentcore config init --interactive

# Output (interactive wizard)
AgentCore Configuration Setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

API Base URL [http://localhost:8001]:
API Timeout (seconds) [30]:
Enable SSL verification [Y/n]:
Default output format [table]: json

Authentication:
Do you have a JWT token? [y/N]: n
Token will be requested on first API call.

✓ Configuration saved to ./.agentcore.yaml

Use 'agentcore config show' to review settings
```

**Command:** `agentcore config show`

```bash
# Usage
agentcore config show [OPTIONS]

# Options
--source TEXT            Show specific source: cli|env|project|global|defaults|merged
--output FORMAT          Output format [default: tree]
--help                   Show help

# Example
$ agentcore config show --source merged

# Output (tree format)
Merged Configuration (5 sources)
├── API
│   ├── base_url: http://localhost:8001 (project config)
│   ├── timeout: 30 (defaults)
│   ├── retries: 3 (defaults)
│   └── verify_ssl: true (defaults)
├── Authentication
│   ├── token: <set> (environment)
│   └── token_file: null (defaults)
└── Output
    ├── format: json (project config)
    ├── color: true (defaults)
    └── timestamps: false (defaults)

Configuration sources (precedence: high → low):
1. CLI arguments
2. Environment variables (AGENTCORE_*)
3. Project config (./.agentcore.yaml) ✓
4. Global config (~/.agentcore/config.yaml)
5. Defaults
```

**Command:** `agentcore config validate`

```bash
# Usage
agentcore config validate [OPTIONS]

# Options
--file PATH              Config file to validate [default: current config]
--help                   Show help

# Example
$ agentcore config validate

# Output
Validating configuration...
   Checking API connectivity... ✓
   Validating authentication token... ✓
   Testing output formats... ✓
   Verifying permissions... ✓

✓ Configuration is valid

API Status:
- Endpoint: http://localhost:8001
- Version: 0.2.0 (A2A Protocol)
- Response time: 45ms
```

### 4.2 Configuration File Format

**Project Config (`.agentcore.yaml`):**

```yaml
# AgentCore CLI Configuration
# Version: 1.0

api:
  base_url: http://localhost:8001
  timeout: 30  # seconds
  retries: 3
  verify_ssl: true

auth:
  token: null  # Use environment variable AGENTCORE_TOKEN
  token_file: null  # Or path to token file

output:
  format: table  # table | json | tree
  color: true
  timestamps: false
  pager: auto  # auto | always | never

logging:
  level: INFO  # DEBUG | INFO | WARNING | ERROR
  file: ~/.agentcore/logs/cli.log
  rotation: 7  # days

defaults:
  agent_list_limit: 50
  task_list_limit: 50
  session_list_limit: 50
```

**Global Config (`~/.agentcore/config.yaml`):**

```yaml
# Global AgentCore Configuration
# Applied to all projects unless overridden

api:
  base_url: https://agentcore.example.com
  verify_ssl: true

auth:
  token_file: ~/.agentcore/token

output:
  format: table
  color: true

# Multiple profiles for different environments
profiles:
  dev:
    api:
      base_url: http://localhost:8001
  staging:
    api:
      base_url: https://staging.agentcore.example.com
  production:
    api:
      base_url: https://api.agentcore.example.com
```

### 4.3 Environment Variables

```bash
# API Configuration
export AGENTCORE_API_BASE_URL="http://localhost:8001"
export AGENTCORE_API_TIMEOUT=30
export AGENTCORE_API_RETRIES=3
export AGENTCORE_API_VERIFY_SSL=true

# Authentication
export AGENTCORE_TOKEN="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..."
export AGENTCORE_TOKEN_FILE="~/.agentcore/token"

# Output
export AGENTCORE_OUTPUT_FORMAT="json"
export AGENTCORE_OUTPUT_COLOR=true

# Logging
export AGENTCORE_LOG_LEVEL="DEBUG"
export AGENTCORE_LOG_FILE="~/.agentcore/logs/cli.log"

# Profile selection
export AGENTCORE_PROFILE="dev"
```

---

## 5. Data Models

### 5.1 Configuration Models

```python
from pydantic import BaseModel, Field, HttpUrl
from typing import Literal, Optional
from pathlib import Path

class ApiConfig(BaseModel):
    """API connection configuration"""
    base_url: HttpUrl = Field(default="http://localhost:8001")
    timeout: int = Field(default=30, ge=1, le=300)
    retries: int = Field(default=3, ge=0, le=10)
    verify_ssl: bool = Field(default=True)

class AuthConfig(BaseModel):
    """Authentication configuration"""
    token: Optional[str] = Field(default=None)
    token_file: Optional[Path] = Field(default=None)

class OutputConfig(BaseModel):
    """Output formatting configuration"""
    format: Literal["table", "json", "tree"] = Field(default="table")
    color: bool = Field(default=True)
    timestamps: bool = Field(default=False)
    pager: Literal["auto", "always", "never"] = Field(default="auto")

class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    file: Path = Field(default=Path.home() / ".agentcore" / "logs" / "cli.log")
    rotation: int = Field(default=7, description="Days to keep logs")

class DefaultsConfig(BaseModel):
    """Default values for CLI operations"""
    agent_list_limit: int = Field(default=50, ge=1, le=1000)
    task_list_limit: int = Field(default=50, ge=1, le=1000)
    session_list_limit: int = Field(default=50, ge=1, le=1000)

class Config(BaseModel):
    """Complete CLI configuration"""
    api: ApiConfig = Field(default_factory=ApiConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
```

### 5.2 JSON-RPC Client Models

```python
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, Union
from uuid import UUID, uuid4

class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 request"""
    jsonrpc: str = Field(default="2.0", const=True)
    method: str = Field(..., description="Method name")
    params: Optional[Dict[str, Any]] = Field(default=None)
    id: Union[str, int, None] = Field(default_factory=lambda: str(uuid4()))

class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 error object"""
    code: int = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    data: Optional[Any] = Field(default=None, description="Additional error data")

class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 response"""
    jsonrpc: str = Field(default="2.0", const=True)
    result: Optional[Any] = Field(default=None)
    error: Optional[JsonRpcError] = Field(default=None)
    id: Union[str, int, None] = Field(...)

    def is_error(self) -> bool:
        return self.error is not None

class RetryConfig(BaseModel):
    """Retry configuration for HTTP requests"""
    max_attempts: int = Field(default=3, ge=1, le=10)
    initial_delay: float = Field(default=1.0, ge=0.1, le=10.0)
    max_delay: float = Field(default=60.0, ge=1.0, le=300.0)
    exponential_base: float = Field(default=2.0, ge=1.0, le=10.0)
    jitter: bool = Field(default=True)
```

### 5.3 Output Models

```python
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime

class TableColumn(BaseModel):
    """Table column definition"""
    key: str
    header: str
    width: Optional[int] = None
    align: Literal["left", "center", "right"] = "left"
    formatter: Optional[str] = None  # e.g., "datetime", "bytes", "percentage"

class TableRow(BaseModel):
    """Table row data"""
    data: Dict[str, Any]

class TableOutput(BaseModel):
    """Table output specification"""
    columns: List[TableColumn]
    rows: List[TableRow]
    title: Optional[str] = None
    footer: Optional[str] = None

class TreeNode(BaseModel):
    """Tree node for hierarchical output"""
    label: str
    value: Optional[str] = None
    children: List["TreeNode"] = []

TreeNode.update_forward_refs()
```

---

## 6. Technology Stack

### 6.1 Runtime & Framework

```yaml
Runtime:
  language: Python 3.12+
  reason: Modern async support, type hints, pattern matching

CLI Framework:
  library: Typer 0.9+
  reason: Type-based CLI generation, auto-completions, rich integration
  alternatives_considered:
    - Click: More verbose, less type-safe
    - argparse: Too low-level, no auto-completion

HTTP Client:
  library: httpx 0.24+
  reason: Async support, HTTP/2, excellent retry support
  alternatives_considered:
    - requests: No async support
    - aiohttp: Complex API, less intuitive

Output Formatting:
  library: Rich 13+
  reason: Beautiful tables, trees, progress bars, syntax highlighting
  alternatives_considered:
    - tabulate: Limited formatting options
    - prettytable: Outdated, poor performance

Configuration:
  library: PyYAML 6+
  reason: Human-readable, widely supported, good ecosystem
  alternatives_considered:
    - TOML: Less flexible for nested structures
    - JSON: Not human-friendly for config files

Validation:
  library: Pydantic 2.0+
  reason: Type-safe validation, JSON schema generation, excellent errors
  alternatives_considered:
    - marshmallow: Less type-safe, more verbose
    - dataclasses: No validation

Authentication:
  library: keyring 24+
  reason: Cross-platform secure token storage
  alternatives_considered:
    - Manual file storage: Security risk
    - Environment variables: Not persistent
```

### 6.2 Development Tools

```yaml
Testing:
  framework: pytest 7.4+
  async: pytest-asyncio 0.21+
  mocking: pytest-mock 3.11+
  coverage: pytest-cov 4.1+

Code Quality:
  linter: ruff 0.1+
  formatter: ruff format
  type_checker: mypy 1.5+ (strict mode)

Package Management:
  tool: uv (fast pip replacement)
  reason: 10-100x faster than pip, built-in venv management

Build & Distribution:
  tool: hatchling (PEP 621)
  reason: Modern, standard-compliant, simple configuration
```

### 6.3 Dependencies

**Core Dependencies:**

```toml
[project]
name = "agentcore-cli"
version = "0.1.0"
requires-python = ">=3.12"

dependencies = [
    "typer[all]>=0.9.0",
    "rich>=13.0.0",
    "httpx>=0.24.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0.0",
    "keyring>=24.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-mock>=3.11.0",
    "pytest-cov>=4.1.0",
    "mypy>=1.5.0",
    "ruff>=0.1.0",
]

[project.scripts]
agentcore = "agentcore_cli.main:app"
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

**Scope:** Test individual modules in isolation

**Coverage Target:** 85%+

**Key Test Cases:**

```python
# tests/unit/test_jsonrpc_client.py

import pytest
from agentcore_cli.client.jsonrpc_client import JsonRpcClient
from agentcore_cli.client.retry import RetryConfig

@pytest.mark.asyncio
async def test_jsonrpc_request_construction():
    """Test JSON-RPC request is properly formatted"""
    client = JsonRpcClient(base_url="http://localhost:8001")
    request = client._build_request("agent.list", {"status": "active"})

    assert request.jsonrpc == "2.0"
    assert request.method == "agent.list"
    assert request.params == {"status": "active"}
    assert request.id is not None

@pytest.mark.asyncio
async def test_jsonrpc_error_handling(mock_httpx):
    """Test JSON-RPC error response handling"""
    mock_httpx.post.return_value.json.return_value = {
        "jsonrpc": "2.0",
        "error": {
            "code": -32000,
            "message": "Agent not found"
        },
        "id": "req-001"
    }

    client = JsonRpcClient(base_url="http://localhost:8001")
    with pytest.raises(AgentNotFoundError) as exc:
        await client.call("agent.get", {"agent_id": "agt_notfound"})

    assert "Agent not found" in str(exc.value)

@pytest.mark.asyncio
async def test_retry_logic_with_exponential_backoff(mock_httpx):
    """Test retry logic with exponential backoff"""
    mock_httpx.post.side_effect = [
        httpx.ConnectError("Connection refused"),
        httpx.ConnectError("Connection refused"),
        httpx.Response(200, json={"jsonrpc": "2.0", "result": {"success": True}, "id": "1"})
    ]

    client = JsonRpcClient(
        base_url="http://localhost:8001",
        retry_config=RetryConfig(max_attempts=3, initial_delay=0.1)
    )

    result = await client.call("agent.list")
    assert result["success"] is True
    assert mock_httpx.post.call_count == 3

# tests/unit/test_config_loader.py

def test_config_precedence():
    """Test configuration precedence: CLI > env > project > global > defaults"""
    # Setup
    os.environ["AGENTCORE_API_BASE_URL"] = "http://env:8001"
    project_config = {"api": {"base_url": "http://project:8001"}}
    global_config = {"api": {"base_url": "http://global:8001"}}
    cli_args = {"api_base_url": "http://cli:8001"}

    # Load merged config
    config = ConfigLoader.load(
        cli_args=cli_args,
        project_config_path=".agentcore.yaml",
        global_config_path="~/.agentcore/config.yaml"
    )

    # CLI args win
    assert config.api.base_url == "http://cli:8001"

def test_config_validation_errors():
    """Test configuration validation with invalid values"""
    with pytest.raises(ValidationError) as exc:
        Config(api=ApiConfig(timeout=-1))  # Invalid negative timeout

    assert "timeout" in str(exc.value)

# tests/unit/test_formatters.py

def test_table_formatter():
    """Test table output formatting"""
    data = [
        {"agent_id": "agt_001", "name": "Agent1", "status": "active"},
        {"agent_id": "agt_002", "name": "Agent2", "status": "inactive"},
    ]

    formatter = TableFormatter(
        columns=[
            TableColumn(key="agent_id", header="Agent ID"),
            TableColumn(key="name", header="Name"),
            TableColumn(key="status", header="Status"),
        ]
    )

    output = formatter.format(data)
    assert "Agent ID" in output
    assert "agt_001" in output
    assert "Agent1" in output

def test_json_formatter():
    """Test JSON output formatting"""
    data = {"agent_id": "agt_001", "name": "Agent1"}
    formatter = JsonFormatter()
    output = formatter.format(data)

    parsed = json.loads(output)
    assert parsed["agent_id"] == "agt_001"
    assert parsed["name"] == "Agent1"

def test_tree_formatter():
    """Test tree output formatting"""
    data = TreeNode(
        label="Agent: agt_001",
        children=[
            TreeNode(label="Name", value="Agent1"),
            TreeNode(label="Status", value="active"),
        ]
    )

    formatter = TreeFormatter()
    output = formatter.format(data)
    assert "Agent: agt_001" in output
    assert "   Name: Agent1" in output
    assert "   Status: active" in output
```

**Testing Tools:**

- pytest for test framework
- pytest-asyncio for async tests
- pytest-mock for mocking
- httpx-mock for HTTP mocking

### 7.2 Integration Tests

**Scope:** Test CLI commands with mocked API responses

**Coverage Target:** All command groups

**Key Test Cases:**

```python
# tests/integration/test_agent_commands.py

import pytest
from typer.testing import CliRunner
from agentcore_cli.main import app

runner = CliRunner()

def test_agent_register_command(mock_api):
    """Test agent register command with mocked API"""
    mock_api.post("/api/v1/jsonrpc").return_json({
        "jsonrpc": "2.0",
        "result": {
            "agent_id": "agt_123",
            "name": "TestAgent",
            "status": "active",
            "created_at": "2025-10-01T12:00:00Z"
        },
        "id": "req-001"
    })

    result = runner.invoke(app, [
        "agent", "register",
        "--name", "TestAgent",
        "--url", "http://localhost:9000",
        "--output", "json"
    ])

    assert result.exit_code == 0
    output = json.loads(result.stdout)
    assert output["agent_id"] == "agt_123"
    assert output["name"] == "TestAgent"

def test_agent_list_command(mock_api):
    """Test agent list command with pagination"""
    mock_api.post("/api/v1/jsonrpc").return_json({
        "jsonrpc": "2.0",
        "result": {
            "agents": [
                {"agent_id": "agt_001", "name": "Agent1", "status": "active"},
                {"agent_id": "agt_002", "name": "Agent2", "status": "active"},
            ],
            "total": 2
        },
        "id": "req-002"
    })

    result = runner.invoke(app, [
        "agent", "list",
        "--status", "active",
        "--limit", "10"
    ])

    assert result.exit_code == 0
    assert "Agent1" in result.stdout
    assert "Agent2" in result.stdout

def test_agent_command_error_handling(mock_api):
    """Test error handling for agent commands"""
    mock_api.post("/api/v1/jsonrpc").return_json({
        "jsonrpc": "2.0",
        "error": {
            "code": -32000,
            "message": "Agent not found"
        },
        "id": "req-003"
    })

    result = runner.invoke(app, [
        "agent", "get", "agt_notfound"
    ])

    assert result.exit_code == 1
    assert "Agent not found" in result.stdout

# tests/integration/test_config_commands.py

def test_config_init_command():
    """Test config init command creates valid config file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / ".agentcore.yaml"

        result = runner.invoke(app, [
            "config", "init",
            "--output", str(config_path)
        ])

        assert result.exit_code == 0
        assert config_path.exists()

        config = yaml.safe_load(config_path.read_text())
        assert "api" in config
        assert "auth" in config

def test_config_show_command():
    """Test config show command displays merged config"""
    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0
    assert "Merged Configuration" in result.stdout
    assert "base_url" in result.stdout
```

### 7.3 End-to-End Tests

**Scope:** Test complete workflows with real AgentCore API

**Coverage Target:** Critical user workflows

**Key Test Cases:**

```python
# tests/e2e/test_full_workflows.py

import pytest
import testcontainers
from agentcore_cli.main import app

@pytest.fixture
def agentcore_api():
    """Start AgentCore API in testcontainer"""
    with testcontainers.compose.DockerCompose(
        filepath="./docker-compose.test.yml"
    ) as compose:
        compose.wait_for("http://localhost:8001/health")
        yield compose

def test_complete_agent_lifecycle(agentcore_api):
    """Test complete agent lifecycle: register -> list -> get -> delete"""
    # 1. Register agent
    result = runner.invoke(app, [
        "agent", "register",
        "--name", "E2ETestAgent",
        "--url", "http://localhost:9000",
        "--output", "json"
    ])
    assert result.exit_code == 0
    agent_data = json.loads(result.stdout)
    agent_id = agent_data["agent_id"]

    # 2. List agents (should include new agent)
    result = runner.invoke(app, ["agent", "list", "--output", "json"])
    assert result.exit_code == 0
    agents = json.loads(result.stdout)
    assert any(a["agent_id"] == agent_id for a in agents)

    # 3. Get agent details
    result = runner.invoke(app, ["agent", "get", agent_id, "--output", "json"])
    assert result.exit_code == 0
    agent = json.loads(result.stdout)
    assert agent["name"] == "E2ETestAgent"

    # 4. Delete agent
    result = runner.invoke(app, ["agent", "delete", agent_id, "--force"])
    assert result.exit_code == 0

    # 5. Verify deletion
    result = runner.invoke(app, ["agent", "get", agent_id])
    assert result.exit_code == 1  # Should fail (not found)

def test_task_creation_and_monitoring(agentcore_api):
    """Test task creation and status monitoring"""
    # 1. Register agent
    result = runner.invoke(app, [
        "agent", "register",
        "--name", "TaskAgent",
        "--url", "http://localhost:9000",
        "--capabilities", "testing",
        "--output", "json"
    ])
    agent_id = json.loads(result.stdout)["agent_id"]

    # 2. Create task
    result = runner.invoke(app, [
        "task", "create",
        "--description", "E2E Test Task",
        "--agent", "testing",
        "--priority", "2",
        "--output", "json"
    ])
    assert result.exit_code == 0
    task_data = json.loads(result.stdout)
    task_id = task_data["task_id"]

    # 3. Monitor task status
    result = runner.invoke(app, ["task", "get", task_id, "--output", "json"])
    assert result.exit_code == 0
    task = json.loads(result.stdout)
    assert task["status"] in ["pending", "running", "completed"]

    # 4. List tasks
    result = runner.invoke(app, ["task", "list", "--output", "json"])
    assert result.exit_code == 0
    tasks = json.loads(result.stdout)
    assert any(t["task_id"] == task_id for t in tasks)
```

**E2E Testing Environment:**

- testcontainers-python for Docker orchestration
- docker-compose.test.yml with AgentCore + PostgreSQL + Redis
- Isolated test database (auto-cleanup)

### 7.4 Performance Tests

**Scope:** Validate performance targets

**Key Metrics:**

- Command response time: <2s for simple commands
- API call overhead: <500ms
- Output rendering: <100ms for 1000 rows
- Config loading: <50ms

```python
# tests/performance/test_command_latency.py

import pytest
import time

def test_agent_list_command_latency(benchmark, mock_api):
    """Benchmark agent list command latency"""
    mock_api.post("/api/v1/jsonrpc").return_json({
        "jsonrpc": "2.0",
        "result": {"agents": [{"agent_id": f"agt_{i}", "name": f"Agent{i}"} for i in range(100)]},
        "id": "req-001"
    })

    def run_command():
        runner.invoke(app, ["agent", "list", "--limit", "100"])

    result = benchmark(run_command)
    assert result < 2.0  # <2s target

def test_table_rendering_performance():
    """Benchmark table rendering for 1000 rows"""
    data = [{"id": i, "name": f"Item{i}", "status": "active"} for i in range(1000)]

    start = time.perf_counter()
    formatter = TableFormatter(columns=[...])
    output = formatter.format(data)
    duration = time.perf_counter() - start

    assert duration < 0.1  # <100ms target
```

### 7.5 Quality Gates

All tests must pass before merge:

```yaml
Required:
  - Unit tests: 85%+ coverage
  - Integration tests: All command groups tested
  - E2E tests: Critical workflows tested
  - Type checking: mypy strict mode (no errors)
  - Linting: ruff (no errors)
  - Performance: All benchmarks pass

Optional (warnings):
  - Documentation coverage
  - Security scan (bandit)
```

---

## 8. Operational Concerns

### 8.1 Logging

**Log Location:** `~/.agentcore/logs/cli.log`

**Log Format:** Structured JSON logs

```json
{
  "timestamp": "2025-10-01T15:30:45.123Z",
  "level": "INFO",
  "command": "agent register",
  "args": {
    "name": "TestAgent",
    "url": "http://localhost:9000"
  },
  "api_method": "agent.register",
  "request_id": "req-001",
  "response_time_ms": 245,
  "status": "success"
}
```

**Log Rotation:**

- Keep last 7 days
- Max file size: 10 MB per day
- Auto-cleanup old logs

**Log Levels:**

```python
DEBUG:   # Verbose output (HTTP requests/responses, config loading)
INFO:    # Normal operations (command execution, API calls)
WARNING: # Recoverable issues (retries, deprecation warnings)
ERROR:   # Failures (API errors, validation failures)
```

### 8.2 Error Handling

**Error Classification:**

```python
# errors.py

class AgentCoreError(Exception):
    """Base exception for all CLI errors"""
    exit_code: int = 1

class ApiError(AgentCoreError):
    """API communication errors"""
    exit_code = 2

class ConfigError(AgentCoreError):
    """Configuration errors"""
    exit_code = 3

class AuthenticationError(AgentCoreError):
    """Authentication errors"""
    exit_code = 4

class ValidationError(AgentCoreError):
    """Input validation errors"""
    exit_code = 5
```

**User-Friendly Error Messages:**

```
# Bad (technical)
Error: JSONRPCError(-32000): Agent not found

# Good (user-friendly)
[Error] Agent not found

The agent 'agt_notfound' does not exist or has been deleted.

Suggestions:
- Use 'agentcore agent list' to see available agents
- Check the agent ID for typos
- Verify you're connected to the correct API endpoint

For more details, run with --debug flag.
```

### 8.3 Monitoring & Telemetry

**Optional Telemetry (Opt-in):**

```yaml
telemetry:
  enabled: false  # Disabled by default
  endpoint: https://telemetry.agentcore.io
  anonymous: true  # No PII collected

metrics:
  - command_usage_frequency
  - command_execution_time
  - error_rates_by_command
  - api_response_times
  - cli_version_distribution
```

**Local Metrics (Always Available):**

```bash
$ agentcore stats

CLI Usage Statistics (Last 7 Days)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Most Used Commands:
1. agent list (45 times)
2. task get (32 times)
3. agent register (12 times)

Average Response Times:
- agent list: 234ms
- task get: 189ms
- session resume: 847ms

Error Rate: 3.2% (8 errors / 247 commands)

Top Errors:
1. API connection timeout (4 times)
2. Agent not found (2 times)
```

### 8.4 Health Checks

**Command:** `agentcore health`

```bash
$ agentcore health

AgentCore Health Check
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ CLI Version: 0.1.0
✓ Configuration: Valid
✓ API Endpoint: http://localhost:8001
✓ API Version: 0.2.0 (A2A Protocol)
✓ API Response Time: 45ms
✓ Authentication: Valid (token expires in 6 days)
✓ Network: Connected

All systems operational
```

### 8.5 Debugging Support

**Debug Mode:**

```bash
# Enable verbose logging
$ agentcore --debug agent list

[DEBUG] Loading configuration...
[DEBUG] Merged config from 5 sources:
  - CLI args: {}
  - Environment: AGENTCORE_API_BASE_URL=http://localhost:8001
  - Project config: ./.agentcore.yaml
  - Global config: ~/.agentcore/config.yaml
  - Defaults: (loaded)

[DEBUG] Building JSON-RPC request:
  Method: agent.list
  Params: {"status": "all", "limit": 50}
  Request ID: req-abc123

[DEBUG] HTTP POST http://localhost:8001/api/v1/jsonrpc
[DEBUG] Request headers: {
  "Content-Type": "application/json",
  "Authorization": "Bearer eyJhbG..."
}
[DEBUG] Request body: {
  "jsonrpc": "2.0",
  "method": "agent.list",
  "params": {...},
  "id": "req-abc123"
}

[DEBUG] API response time: 234ms
[DEBUG] Response status: 200 OK
[DEBUG] Response body: {
  "jsonrpc": "2.0",
  "result": {...},
  "id": "req-abc123"
}

[INFO] Command completed successfully in 312ms
```

**Trace Mode:**

```bash
# Full HTTP request/response tracing
$ agentcore --trace agent list

> POST /api/v1/jsonrpc HTTP/1.1
> Host: localhost:8001
> User-Agent: agentcore-cli/0.1.0
> Content-Type: application/json
> Authorization: Bearer eyJhbG...
> Content-Length: 156
>
> {"jsonrpc":"2.0","method":"agent.list","params":{"status":"all"},"id":"req-001"}

< HTTP/1.1 200 OK
< Content-Type: application/json
< Content-Length: 543
< Server: uvicorn
< Date: Wed, 01 Oct 2025 15:30:45 GMT
<
< {"jsonrpc":"2.0","result":{...},"id":"req-001"}
```

---

## 9. Security Considerations

### 9.1 Authentication Token Storage

**Default: System Keychain (Secure)**

```python
import keyring

# Store token securely
keyring.set_password("agentcore", "default", jwt_token)

# Retrieve token
token = keyring.get_password("agentcore", "default")
```

**Platforms:**

- macOS: Keychain Access
- Windows: Windows Credential Manager
- Linux: Secret Service API (GNOME Keyring, KWallet)

**Fallback: Plaintext File (with warning)**

```bash
$ agentcore auth login

Warning: System keychain not available.
Token will be stored in plaintext at:
  ~/.agentcore/token

This is insecure. Anyone with access to this file can impersonate you.

Continue? [y/N]:
```

### 9.2 HTTPS Enforcement

```python
# Validate SSL by default
if config.api.base_url.startswith("http://"):
    console.print("[yellow]Warning: Using insecure HTTP connection[/yellow]")
    console.print("Consider using HTTPS for production use")

# Allow disabling SSL verification (with warning)
if not config.api.verify_ssl:
    console.print("[red]Warning: SSL verification disabled[/red]")
    console.print("This is insecure and should only be used for testing")
```

### 9.3 Input Validation

```python
# Sanitize all user inputs
@register_jsonrpc_method("agent.register")
async def register_agent(name: str, url: HttpUrl, ...):
    # Pydantic validates HttpUrl format
    # Additional validation
    if not url.scheme in ["http", "https"]:
        raise ValidationError("URL must use http or https scheme")

    # Sanitize agent name (prevent injection)
    name = name.strip()
    if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", name):
        raise ValidationError("Agent name must be alphanumeric (1-64 chars)")
```

### 9.4 Secrets in Logs

```python
# Redact sensitive data in logs
def sanitize_for_logging(data: Dict) -> Dict:
    """Remove sensitive fields from log data"""
    sensitive_keys = ["token", "password", "secret", "api_key"]

    sanitized = data.copy()
    for key in sensitive_keys:
        if key in sanitized:
            sanitized[key] = "<redacted>"

    return sanitized
```

---

## 10. Risk Analysis

### 10.1 Risk Matrix

| Risk ID | Risk | Impact | Likelihood | Mitigation | Contingency |
|---------|------|--------|------------|------------|-------------|
| CLI-R1 | JSON-RPC client complexity causes bugs | High | Medium | Comprehensive unit tests, use proven patterns from research | Simplify to single requests only (no batch) |
| CLI-R2 | Session API not ready (A2A-019, A2A-020 dependency) | High | Medium | Mock session API for CLI development, integration tests | Defer session commands to Sprint 2 |
| CLI-R3 | Configuration precedence bugs | Medium | High | Extensive unit tests for all 5 levels, clear documentation | Simplify to 3 levels (CLI > env > defaults) |
| CLI-R4 | Rich library performance issues with large datasets | Low | Low | Pagination support, benchmark with 1000+ rows | Fallback to simple text output |
| CLI-R5 | Token storage security issues | Medium | Medium | Use system keychain by default, warn on plaintext fallback | Manual token refresh workflow |
| CLI-R6 | Cross-platform compatibility (macOS, Linux, Windows) | Medium | Medium | Test on all 3 platforms, use pathlib for paths | Document platform-specific issues |
| CLI-R7 | API versioning and compatibility | Medium | Low | Version negotiation in client, graceful degradation | Fail fast with clear error message |

### 10.2 Critical Path Risks

**Blocker: Session API Dependency (CLI-R2)**

Tasks CLI-004 (session save), CLI-005 (session resume) depend on:

- A2A-019: Session Snapshot Creation (5 SP, Week 9)
- A2A-020: Session Resumption (5 SP, Week 9-10)
- A2A-021: Session Management API (3 SP, Week 10)

**Mitigation:**

1. Implement session commands with mock API first
2. Integration tests with testcontainers when API ready
3. Parallel track: Focus on agent/task commands (no dependency)

**Impact:** 8 SP out of 34 SP (24%) blocked, but not on critical path

### 10.3 Technical Debt

**Known Issues:**

1. No real-time event streaming (basic polling only)
2. Limited output paging for large datasets
3. No interactive TUI (future enhancement)
4. Shell completions generated but not comprehensive

**Planned Improvements:**

- Phase 2: WebSocket support for real-time updates
- Phase 2: Interactive mode with prompt_toolkit
- Phase 3: Advanced shell completions with examples

---

## 11. Development Workflow

### 11.1 Setup

```bash
# Clone repository
git clone https://github.com/your-org/agentcore.git
cd agentcore/cli

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies with uv
uv pip install -e ".[dev]"

# Verify installation
agentcore --version

# Run tests
uv run pytest

# Type check
uv run mypy src/

# Lint
uv run ruff check src/ tests/
```

### 11.2 Development Cycle

```bash
# 1. Create feature branch
git checkout -b feature/cli-session-commands

# 2. Implement feature with TDD
uv run pytest tests/unit/test_session_commands.py --watch

# 3. Run full test suite
uv run pytest --cov

# 4. Type check
uv run mypy src/

# 5. Lint and format
uv run ruff check --fix src/ tests/
uv run ruff format src/ tests/

# 6. Commit
git add .
git commit -m "feat(cli): add session save/resume commands

- Implement session.save CLI command
- Implement session.resume CLI command
- Add integration tests with mock API
- Update documentation

Refs: CLI-004, CLI-005"

# 7. Push and create PR
git push origin feature/cli-session-commands
```

### 11.3 CI/CD Pipeline

```yaml
# .github/workflows/cli-tests.yml

name: CLI Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.12", "3.13"]

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install uv
        run: pip install uv

      - name: Install dependencies
        run: uv pip install -e ".[dev]"

      - name: Run tests
        run: uv run pytest --cov --cov-report=xml

      - name: Type check
        run: uv run mypy src/

      - name: Lint
        run: uv run ruff check src/ tests/

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  e2e:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: uv pip install -e ".[dev]"

      - name: Start AgentCore API
        run: docker compose -f docker-compose.test.yml up -d

      - name: Wait for API
        run: ./scripts/wait-for-api.sh

      - name: Run E2E tests
        run: uv run pytest tests/e2e/

      - name: Cleanup
        run: docker compose -f docker-compose.test.yml down
```

---

## 12. Implementation Checklist

### Phase 1: Core CLI Framework (Week 1, 8 SP)

- [ ] **CLI-001: Project Setup and Typer Integration** (2 SP)
  - [ ] Initialize CLI package structure
  - [ ] Set up Typer app with basic commands
  - [ ] Configure Rich console
  - [ ] Add shell completion support
  - [ ] Write unit tests for CLI initialization

- [ ] **CLI-002: JSON-RPC Client Implementation** (3 SP)
  - [ ] Implement JSON-RPC 2.0 request/response models
  - [ ] Build async HTTP client with httpx
  - [ ] Add retry logic with exponential backoff
  - [ ] Implement error mapping (JSON-RPC → CLI exceptions)
  - [ ] Write unit tests for client (90%+ coverage)

- [ ] **CLI-003: Configuration System** (3 SP)
  - [ ] Define Pydantic config models
  - [ ] Implement multi-level config loader (5 levels)
  - [ ] Add YAML parsing and validation
  - [ ] Implement config precedence logic
  - [ ] Write unit tests for all precedence scenarios

**Phase 1 Exit Criteria:**

- Typer CLI running with `agentcore --help`
- JSON-RPC client passing all unit tests
- Configuration loading from all 5 sources

### Phase 2: Agent & Task Commands (Week 2, 13 SP)

- [ ] **CLI-006: Agent Commands** (5 SP)
  - [ ] Implement `agentcore agent register`
  - [ ] Implement `agentcore agent list`
  - [ ] Implement `agentcore agent get`
  - [ ] Implement `agentcore agent delete`
  - [ ] Add output formatters (table, JSON, tree)
  - [ ] Write integration tests with mock API

- [ ] **CLI-007: Task Commands** (5 SP)
  - [ ] Implement `agentcore task create`
  - [ ] Implement `agentcore task get`
  - [ ] Implement `agentcore task list`
  - [ ] Implement `agentcore task update`
  - [ ] Add task artifact handling
  - [ ] Write integration tests with mock API

- [ ] **CLI-008: Authentication** (3 SP)
  - [ ] Implement JWT token storage (keyring)
  - [ ] Add `agentcore auth login` command
  - [ ] Add `agentcore auth logout` command
  - [ ] Add `agentcore auth status` command
  - [ ] Handle token refresh
  - [ ] Write unit tests for auth manager

**Phase 2 Exit Criteria:**

- All agent and task commands working
- Authentication flow complete
- 85%+ test coverage

### Phase 3: Session & Workflow Commands (Week 3, 8 SP)

- [ ] **CLI-004: Session Commands** (3 SP)
  - [ ] Implement `agentcore session save`
  - [ ] Implement `agentcore session resume`
  - [ ] Implement `agentcore session list`
  - [ ] Implement `agentcore session info`
  - [ ] Write integration tests with mock session API
  - [ ] **Dependency:** A2A-019, A2A-020, A2A-021 (Phase 5)

- [ ] **CLI-009: Workflow Commands** (5 SP)
  - [ ] Implement `agentcore workflow create`
  - [ ] Implement `agentcore workflow execute`
  - [ ] Implement `agentcore workflow status`
  - [ ] Add workflow definition YAML parsing
  - [ ] Add real-time execution monitoring (polling)
  - [ ] Write integration tests with mock API

**Phase 3 Exit Criteria:**

- Session commands implemented (may use mock API)
- Workflow commands working
- Integration tests passing

### Phase 4: Polish & Documentation (Week 4, 5 SP)

- [ ] **CLI-010: Config Commands** (2 SP)
  - [ ] Implement `agentcore config init`
  - [ ] Implement `agentcore config show`
  - [ ] Implement `agentcore config validate`
  - [ ] Add interactive config wizard
  - [ ] Write integration tests

- [ ] **CLI-011: Output Formatters & Error Handling** (2 SP)
  - [ ] Polish table formatting (Rich)
  - [ ] Add pagination for large datasets
  - [ ] Improve error messages (user-friendly)
  - [ ] Add color themes
  - [ ] Write unit tests for formatters

- [ ] **CLI-012: Documentation & Examples** (1 SP)
  - [ ] Write CLI usage guide
  - [ ] Add command examples
  - [ ] Create tutorial for common workflows
  - [ ] Document configuration options
  - [ ] Add troubleshooting guide

**Phase 4 Exit Criteria:**

- All 12 tasks complete (34 SP)
- 85%+ test coverage
- Documentation complete
- Ready for beta release

---

## 13. References

### 13.1 Research Sources

**Typer CLI Best Practices:**

- Testing patterns: Use `CliRunner` for invocation tests, pytest for framework
- Error handling: Try-except blocks, custom exceptions, informative messages
- Design patterns: Command pattern, dependency injection via type hints, stateless commands
- Anti-patterns: Avoid hard-coded values, always handle errors gracefully

**Rich Library Performance:**

- Tables render efficiently for small-medium datasets (<1000 rows)
- For large datasets: Use pagination, lazy rendering, or fallback to simple text
- Column width optimization: Let Rich calculate optimal widths automatically

**JSON-RPC 2.0 Client:**

- Retry logic: Exponential backoff with jitter, conditional retries for network errors
- Error handling: Use optional `data` field for additional error context
- Libraries: `jsonrpcclient` for client, `tenacity` for retry logic
- Best practices: Idempotent requests, proper timeout handling, error-specific handling

### 13.2 Related Documentation

- `/Users/druk/WorkSpace/AetherForge/AgentCore/docs/specs/cli-layer/spec.md`
- `/Users/druk/WorkSpace/AetherForge/AgentCore/docs/specs/cli-layer/plan.md`
- `/Users/druk/WorkSpace/AetherForge/AgentCore/docs/specs/cli-layer/tasks.md`
- `/Users/druk/WorkSpace/AetherForge/AgentCore/docs/breakdown/a2a-protocol.md`
- Google A2A Protocol v0.2 Specification

### 13.3 External Dependencies

**AgentCore A2A Protocol API:**

- JSON-RPC 2.0 endpoint: `POST /api/v1/jsonrpc`
- WebSocket endpoint: `WS /ws` (future)
- Health check: `GET /health`

**Session Management API (Phase 5 Dependency):**

- `session.save` - A2A-019 (Week 9)
- `session.resume` - A2A-020 (Week 9-10)
- `session.list`, `session.info` - A2A-021 (Week 10)

---

## 14. Appendix

### 14.1 Glossary

- **JSON-RPC 2.0**: Stateless, lightweight remote procedure call protocol encoded in JSON
- **A2A Protocol**: Agent2Agent protocol by Google for agent discovery and communication
- **Typer**: Python CLI framework based on type hints
- **Rich**: Python library for rich text and beautiful formatting in terminal
- **keyring**: Python library for accessing system keychain for secure credential storage
- **httpx**: Modern async HTTP client for Python
- **testcontainers**: Library for running Docker containers in tests

### 14.2 Command Quick Reference

```bash
# Agent Management
agentcore agent register --name NAME --url URL [--capabilities CAPS]
agentcore agent list [--status STATUS] [--limit N]
agentcore agent get <agent_id>
agentcore agent delete <agent_id> [--force]

# Task Management
agentcore task create --description DESC --agent AGENT [--priority N]
agentcore task get <task_id> [--show-artifacts]
agentcore task list [--status STATUS] [--agent AGENT]

# Session Management (Phase 5 Dependency)
agentcore session save --name NAME [--description DESC] [--tags TAGS]
agentcore session resume <session_id> [--replay-events]
agentcore session list [--state STATE]

# Workflow Management
agentcore workflow create --file FILE
agentcore workflow execute <workflow_id> [--watch]

# Configuration
agentcore config init [--global] [--interactive]
agentcore config show [--source SOURCE]
agentcore config validate

# Authentication
agentcore auth login
agentcore auth logout
agentcore auth status

# Utility
agentcore health
agentcore stats
agentcore --version
agentcore --help
```

### 14.3 Exit Codes

```text
0  - Success
1  - General error (default)
2  - API communication error
3  - Configuration error
4  - Authentication error
5  - Validation error
130 - Interrupted by user (Ctrl+C)
```

---

**End of Technical Breakdown**

**Next Steps:**

1. Review and approve this breakdown
2. Begin Phase 1 implementation (Week 1)
3. Set up CI/CD pipeline
4. Schedule daily standups

**Questions/Clarifications:**

- Session API timeline confirmation (Phase 5 dependency)
- Cross-platform testing resources (macOS, Linux, Windows)
- Beta release target date

**Document Version:** 1.0
**Last Updated:** 2025-10-01
**Status:** Ready for Implementation
